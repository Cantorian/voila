import asyncio
import json
from typing import Awaitable, Union
from jupyter_server.base.handlers import JupyterHandler
from tornado.websocket import WebSocketHandler
from tornado.web import HTTPError
from tornado.ioloop import IOLoop

try:
    JUPYTER_SERVER_2 = True
    from jupyter_server.base.websocket import WebSocketMixin
except ImportError:
    JUPYTER_SERVER_2 = False
from jupyter_core.utils import ensure_async
from nbclient.exceptions import CellExecutionError
from voila.execute import VoilaExecutor, strip_code_cell_warnings
import nbformat
import traceback
import sys

if JUPYTER_SERVER_2:
    # Module-level lock for thread-safe access to shared execution data
    _execution_data_lock = asyncio.Lock()

    class ExecutionRequestHandler(WebSocketMixin, WebSocketHandler, JupyterHandler):
        _execution_data = {}
        _execution_data_lock = _execution_data_lock

        def initialize(self, **kwargs):
            super().initialize()
            self.ping_interval = 120 * 1000  # 120 seconds
            self.ping_timeout = 20 * 1000  # 20 seconds
            self.max_message_size = 200 * 1024 * 1024  # 200 MB

        async def open(self, kernel_id: str) -> None:
            """Create a new websocket connection, this connection is
            identified by the kernel id.

            Args:
                kernel_id (str): Kernel id used by the notebook when it opens
                the websocket connection.
            """
            identity_provider = self.settings.get("identity_provider")
            user = await ensure_async(identity_provider.get_user(self))
            if user is None:
                raise HTTPError(403, "Unauthenticated")
            super().open()

            self._kernel_id = kernel_id
            self.write_message({"action": "initialized", "payload": {}})

        async def on_message(
            self, message_str: Union[str, bytes]
        ) -> Union[Awaitable[None], None]:
            message = json.loads(message_str)
            action = message.get("action", None)
            payload = message.get("payload", {})

            if action != "execute":
                self.log.info(f"Unknown action received: {action}")
                await ensure_async(self.write_message(
                    {
                        "action": "error",
                        "payload": {"error": f"Unknown action: {action}"},
                    }
                ))
                return

            request_kernel_id = payload.get("kernel_id")
            if request_kernel_id != self._kernel_id:
                await self.write_message(
                    {
                        "action": "execution_error",
                        "payload": {"error": "Kernel ID does not match"},
                    }
                )
                return

            km = await ensure_async(self.kernel_manager.get_kernel(self._kernel_id))
            
            # Safely retrieve execution data with lock to prevent race conditions
            async with self._execution_data_lock:
                execution_data = self._execution_data.pop(self._kernel_id, None)
            
            if execution_data is None:
                await self.write_message(
                    {
                        "action": "execution_error",
                        "payload": {"error": "Missing notebook data"},
                    }
                )
                return

            nb = execution_data["nb"]
            self._executor = executor = VoilaExecutor(
                nb,
                km=km,
                config=execution_data["config"],
                show_tracebacks=execution_data["show_tracebacks"],
            )
            executor.kc = await executor.async_start_new_kernel_client()

            # schedule notebook run and return so we don't block other messages
            self._notebook_task = asyncio.create_task(self._run_notebook(nb, executor))
            return

        async def _run_notebook(self, nb, executor) -> Union[Awaitable[None], None]:
            try:
                total_cell = len(nb.cells)
                for cell_idx, input_cell in enumerate(nb.cells):
                    # Check for cancellation
                    if asyncio.current_task().cancelled():
                        break
                    
                    try:
                        output_cell = await executor.execute_cell(
                            input_cell, None, cell_idx, store_history=False
                        )
                    except TimeoutError:
                        output_cell = input_cell

                    except CellExecutionError:
                        self.log.exception(
                            "Error at server while executing cell: %r", input_cell
                        )
                        if executor.should_strip_error():
                            strip_code_cell_warnings(input_cell)
                            executor.strip_code_cell_errors(input_cell)
                        output_cell = input_cell

                    except Exception as e:
                        self.log.exception(
                            "Error at server while executing cell: %r", input_cell
                        )
                        output_cell = nbformat.v4.new_code_cell()
                        if executor.should_strip_error():
                            output_cell.outputs = [
                                {
                                    "output_type": "stream",
                                    "name": "stderr",
                                    "text": "An exception occurred at the server (not the notebook). {}".format(
                                        executor.cell_error_instruction
                                    ),
                                }
                            ]
                        else:
                            output_cell.outputs = [
                                {
                                    "output_type": "error",
                                    "ename": type(e).__name__,
                                    "evalue": str(e),
                                    "traceback": traceback.format_exception(
                                        *sys.exc_info()
                                    ),
                                }
                            ]
                    finally:
                        output_cell.pop("source", None)
                        try:
                            await asyncio.wait_for(
                                self.write_message(
                                    {
                                        "action": "execution_result",
                                        "payload": {
                                            "output_cell": output_cell,
                                            "cell_index": cell_idx,
                                            "total_cell": total_cell,
                                        },
                                    }
                                ),
                                timeout=10.0
                            )
                        except asyncio.TimeoutError:
                            self.log.warning("Timeout sending execution result for cell %d", cell_idx)
                        except Exception as e:
                            self.log.exception("Error sending execution result for cell %d: %s", cell_idx, e)
            except asyncio.CancelledError:
                self.log.info("Notebook execution cancelled")
                raise
            finally:
                # Cleanup executor
                if executor and executor.kc:
                    try:
                        await ensure_async(executor.kc.stop_channels())
                    except Exception as e:
                        self.log.exception("Error stopping kernel channels: %s", e)

        def on_close(self) -> None:
            # Cancel any running notebook execution
            if hasattr(self, '_notebook_task') and self._notebook_task and not self._notebook_task.done():
                self._notebook_task.cancel()
            
            # Stop kernel channels
            if self._executor and self._executor.kc:
                try:
                    asyncio.create_task(ensure_async(self._executor.kc.stop_channels()))
                except Exception as e:
                    self.log.exception("Error stopping kernel channels on close: %s", e)

else:

    class ExecutionRequestHandler:
        pass

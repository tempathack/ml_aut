from functools import wraps
from datetime import datetime
import multiprocessing
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    END = '\033[0m'  # Reset color to default


class WrapStack(object):
    def __int__(self,workers):
        self.Logger_dict = {}

    def FUNCTION_SCREEN(func):
        @wraps(func)
        def _impl(self, *args, **kwargs):
            worker_id = multiprocessing.current_process().name

            start=datetime.now()
            WrapStack.write_log(f'Started Execution of Function {Colors.CYAN}{func.__name__.upper()}{Colors.END}: kwargs[{str(kwargs)}]',message_type='task_normal' if worker_id =='MainProcess' else 'task_parallel',worker_id=worker_id)
            res = func(self, *args, **kwargs)
            end=datetime.now()
            WrapStack.write_log(f'Finished Execution of Function {Colors.CYAN}{func.__name__.upper()}{Colors.END}: kwargs[{str(kwargs)}]',message_type='task_normal' if worker_id =='MainProcess' else 'task_parallel',worker_id=worker_id)
            duration=end-start
            WrapStack.write_log(f'{Colors.CYAN}{func.__name__.upper()}{Colors.END} Duration {duration.total_seconds()}: kwargs[{str(kwargs)}]',message_type='time',worker_id=worker_id)

            return res

        return _impl
    @staticmethod
    def write_log(text:str,message_type:str,worker_id=None):
        end=Colors.END

        if message_type=='task_normal':
            color=Colors.GREEN
            text_ =f"{color}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}|SEQUENTIAL]{end} "+text
        elif message_type=='task_parallel':
            color=Colors.MAGENTA
            text_ =f"{color}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}|PARALLEL({worker_id})]{end} "+text
        elif message_type=='time':
            color=Colors.BLUE
            text_ =f"{color}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}|ELAPSED ({worker_id if worker_id is not None else 'Main'})]{end} "+text
        elif message_type=='warning':
            color=Colors.YELLOW
            text_=f"{color}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}|CAUTION]{end}   "+text

        print(text_,flush=True)
        #self.Logger_dict[f'{datetime.now()}']=text_
    FUNCTION_SCREEN=staticmethod(FUNCTION_SCREEN)
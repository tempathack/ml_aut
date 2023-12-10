from functools import wraps
from datetime import datetime

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    END = '\033[0m'  # Reset color to default

MULT_FLG='SEQUENTIAL'
class WrapStack(object):
    def __int__(self,workers):
        self.Logger_dict = {}

    def FUNCTION_SCREEN(func):
        @wraps(func)
        def _impl(self, *args, **kwargs):
            start=datetime.now()
            WrapStack.write_log(f'Started Execution of Function {func.__name__}:[{str(kwargs)}]',message_type='task_normal' if MULT_FLG=='SEQUENTIAL' else 'task_parallel')
            res = func(self, *args, **kwargs)
            end=datetime.now()
            WrapStack.write_log(f'Finished Execution of Function {func.__name__}:[{str(kwargs)}]',message_type='task_normal' if MULT_FLG=='SEQUENTIAL' else 'task_parallel')
            duration=end-start
            WrapStack.write_log(f'{func.__name__} Duration {duration.total_seconds()}:[{str(kwargs)}]',message_type='time')

            return res

        return _impl
    @staticmethod
    def write_log(text:str,message_type:str):
        end=Colors.END

        if message_type=='task_normal':
            color=Colors.GREEN
            text_ =f"{color}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}|SEQUENTIAL]{end} "+text
        elif message_type=='task_parallel':
            color=Colors.BLUE
            text_ =f"{color}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}|PARALLEL]{end} "+text
        elif message_type=='time':
            color=Colors.BLUE
            text_ =f"{color}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}|ELAPSED]{end}    "+text
        elif message_type=='warning':
            color=Colors.YELLOW
            text_=f"{color}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}|CAUTION]{end}   "+text

        print(text_,flush=True)
        #self.Logger_dict[f'{datetime.now()}']=text_
    FUNCTION_SCREEN=staticmethod(FUNCTION_SCREEN)
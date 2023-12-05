from functools import wraps



class WRAPSTACK(object):
    'Class Method to Track all Functions Executed and to obtain Error Handling/Execution time'
    Logger_dict = {}


    def FUNCTION_SCREEN(func):
        @wraps(func)
        def _impl(self, *args, **kwargs):

            res = func(self, *args, **kwargs)

            return res

        return _impl

    FUNCTION_SCREEN=staticmethod(FUNCTION_SCREEN)
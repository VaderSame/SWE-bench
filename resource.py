# resource.py
def getrlimit(*args, **kwargs): return (0, 0)
def setrlimit(*args, **kwargs): pass
def getpagesize(*args, **kwargs): return 4096
def getrusage(*args, **kwargs): return None

RLIMIT_AS = 0
RLIMIT_CORE = 1
RLIMIT_CPU = 2
RLIMIT_DATA = 3
RLIMIT_FSIZE = 4
RLIMIT_MEMLOCK = 5
RLIMIT_NOFILE = 6
RLIMIT_NPROC = 7
RLIMIT_RSS = 8
RLIMIT_STACK = 9
RUSAGE_CHILDREN = 1
RUSAGE_SELF = 0
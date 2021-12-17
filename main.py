from options import Options
from engine import NiqaEngine


if __name__ == '__main__':
    opt = Options().parse()
    engine = NiqaEngine(opt)
    engine()





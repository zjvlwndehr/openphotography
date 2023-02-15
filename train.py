from pathlib import Path

from engine import *

if __name__ == '__main__':
    engine = Engine(Path('dataset/positive'), Path('data/positive'))
    engine.train()
else:
    assert False, '[INFO]: This file is not intended to be imported.\n\tTo use it, execute it directly.'
    
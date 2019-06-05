import sys

from lib import task
from lib.domains.misc import algolispPrimitives, deepcoderPrimitives
from lib.domains.misc import RobustFillPrimitives
from lib.domains.misc import napsPrimitives
from lib.domains.tower import makeTowerTasks
from lib.domains.tower import towerPrimitives
from lib.domains.regex import regexPrimitives
from lib.domains.regex import makeRegexTasks
from lib.domains.logo import logoPrimitives
from lib.domains.logo import makeLogoTasks
from lib.domains.list import listPrimitives
from lib.domains.list import makeListTasks
from lib.domains.arithmetic import arithmeticPrimitives
from lib.domains.text import textPrimitives
from lib.domains.text import makeTextTasks

# TODO: commented out due to circular import error
# from lib.domains.puddleworld import makePuddleworldTasks
# from lib.domains.puddleworld import puddleworldPrimitives

from lib import primitiveGraph

# Required for backwards-compatibility with old pickle files
# from before the EC codebase refactor:
sys.modules['task'] = task
sys.modules['algolispPrimitives'] = algolispPrimitives
sys.modules['RobustFillPrimitives'] = RobustFillPrimitives
sys.modules['napsPrimitives'] = napsPrimitives
sys.modules['makeTowerTasks'] = makeTowerTasks
sys.modules['towerPrimitives'] = towerPrimitives
sys.modules['regexPrimitives'] = regexPrimitives
sys.modules['makeRegexTasks'] = makeRegexTasks
sys.modules['deepcoderPrimitives'] = deepcoderPrimitives
sys.modules['logoPrimitives'] = logoPrimitives
sys.modules['makeLogoTasks'] = makeLogoTasks
sys.modules['listPrimitives'] = listPrimitives
sys.modules['makeListTasks'] = makeListTasks
sys.modules['arithmeticPrimitives'] = arithmeticPrimitives
sys.modules['textPrimitives'] = textPrimitives
sys.modules['makeTextTasks'] = makeTextTasks
# sys.modules['makePuddleworldTasks'] = makePuddleworldTasks
# sys.modules['puddleworldPrimitives'] = puddleworldPrimitives
sys.modules['primitiveGraph'] = primitiveGraph

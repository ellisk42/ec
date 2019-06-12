"""
EC codebase Python library (AKA the "frontend")

Module mapping details:

TODO: remove module mapping code when backwards-compatibility is no longer required.

The below module mapping is required for backwards-compatibility with old pickle files
generated from before the EC codebase refactor. New files added to the codebase do not
need to be added to the mapping, but if the existing modules are moved, then this the
mapping needs to be updated to reflect the move or rename.

The mapping uses the following pattern:

    sys.modules[<old module path>] = <new module reference>

This is because the previous structure of the codebase was completely flat, and when refactoring
to a hierarchical files, loading previous pickle files no longer works properly. It is important
to retain the ability to read old pickle files generated from official experiments. As a workaround,
the old module paths are included below. A preferable alternative would be to export program state
into JSON files instead of pickle files to avoid issues where the underlying classes change, so that
could be a future improvement to this project. Until then, we use the module mapping workaround.

For more info, see this StackOverflow answer: https://stackoverflow.com/a/2121918/2573242
"""
import sys

from lib import differentiation
from lib import ec
from lib import enumeration
from lib import fragmentGrammar
from lib import fragmentUtilities
from lib import frontier
from lib import grammar
from lib import likelihoodModel
from lib import program
from lib import primitiveGraph
from lib import recognition
from lib import task
from lib import taskBatcher
from lib import type
from lib import utilities
from lib import vs
from lib.domains.misc import algolispPrimitives, deepcoderPrimitives
from lib.domains.misc import RobustFillPrimitives
from lib.domains.misc import napsPrimitives
from lib.domains.tower import makeTowerTasks
from lib.domains.tower import towerPrimitives
from lib.domains.tower import tower_common
from lib.domains.tower import main as tower_main
from lib.domains.regex import groundtruthRegexes
from lib.domains.regex import regexPrimitives
from lib.domains.regex import makeRegexTasks
from lib.domains.logo import logoPrimitives
from lib.domains.logo import makeLogoTasks
from lib.domains.list import listPrimitives
from lib.domains.list import makeListTasks
from lib.domains.arithmetic import arithmeticPrimitives
from lib.domains.text import textPrimitives
from lib.domains.text import makeTextTasks

sys.modules['differentiation'] = differentiation
sys.modules['ec'] = ec
sys.modules['enumeration'] = enumeration
sys.modules['fragmentGrammar'] = fragmentGrammar
sys.modules['fragmentUtilities'] = fragmentUtilities
sys.modules['frontier'] = frontier
sys.modules['grammar'] = grammar
sys.modules['likelihoodModel'] = likelihoodModel
sys.modules['program'] = program
sys.modules['recognition'] = recognition
sys.modules['task'] = task
sys.modules['taskBatcher'] = taskBatcher
sys.modules['type'] = type
sys.modules['utilities'] = utilities
sys.modules['vs'] = vs
sys.modules['algolispPrimitives'] = algolispPrimitives
sys.modules['RobustFillPrimitives'] = RobustFillPrimitives
sys.modules['napsPrimitives'] = napsPrimitives
sys.modules['makeTowerTasks'] = makeTowerTasks
sys.modules['towerPrimitives'] = towerPrimitives
sys.modules['tower_common'] = tower_common
sys.modules['tower'] = tower_main
sys.modules['groundtruthRegexes'] = groundtruthRegexes
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
sys.modules['primitiveGraph'] = primitiveGraph

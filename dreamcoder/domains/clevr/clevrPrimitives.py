from dreamcoder.program import Primitive, Program
from dreamcoder.type import *
from dreamcoder.domains.list.listPrimitives import _eq, _map, _fold, _if, _car, _cdr, _cons, _gt, _not, _isEmpty, _lt

"""
clevrPrimitives.py | Author: Catherine Wong
Contains primitives designed to operate on symbolic CLEVR scene graph reasoning questions.

This contains a reimplementation of the DSL used in the original CLEVR paper, and a set of LISP-based list manipulation primitives from which we can bootstrap these primitives.

CLEVR scenes are sorted lists of Objects of the following form:
    list(Object) = [{
        "id": unique ID,
        "color", "size", "shape", "material": string attributes,
        "left", "right", "front", "behind": [list of integers for object IDs to the <RELATION> of the current object.]
    }]
CLEVR tasks have examples of the form:
    Array of [ ([x], y)]    # Where x is a scene, and y is an answer object.
"""
# A 'null object' to allow safe operations that remove an object from an already empty list.
NULL_STRING = "NULL"
NULL_COLOR = f"{NULL_STRING}_COLOR"
NULL_SIZE = f"{NULL_STRING}_SIZE"
NULL_SHAPE = f"{NULL_STRING}_SHAPE"
NULL_MATERIAL = f"{NULL_STRING}_MATERIAL"

NULL_OBJECT = {   "id" : -1,
        "color" : NULL_COLOR, 
        "size" : NULL_SIZE,
        "shape" : NULL_SHAPE,
        "material" : NULL_MATERIAL,
        "left" : [],
        "right" : [],
        "front" : [],
        "behind" : [],
    }

# Utilities for sorting and deduplicating objects.
def sort_objs(obj_list):
    return sorted(obj_list, key=lambda o: o["id"])
    
# Sort and deduplicate. The final scene equality requires absolute list equality, so we use a sort and deduplicate after all list operations to preserve this.
def sort_and_dedup_obj_list(obj_list):
    seen = set()
    deduped = []
    for o in obj_list:
        if o["id"] not in seen:
            seen.add(o["id"])
            deduped.append(o)
    return sort_objs(deduped)

"""String constant values"""
attribute_constants = {
    'color' : ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"],
    'relation' : ["left", "right", "behind", "front"],
    'shape' : ["cube", "sphere", "cylinder"],
    'size' : ["small", "large"],
    'material' : ["rubber", "metal"]
}
def clevr_lexicon():
    """Defines the token lexicon for all CLEVR I/O examples."""
    lexicon = set()
    for k, values in attribute_constants.items():
        lexicon.add(k)
        lexicon.update(values)
    lexicon.update([str(val) for val in range(-1, 100)]) # Since we could accidentally count something large.
    lexicon.update([str(val) for val in (True, False)])
    lexicon.add("id")
    lexicon.update([NULL_COLOR, NULL_SIZE, NULL_SHAPE, NULL_MATERIAL])
    return list(lexicon)

"""Base Types: in addition to these, CLEVR answers also use the Boolean, integer, and List(ClevrObject) types."""
tclevrcolor = baseType("tclevrcolor")
tclevrsize = baseType("tclevrsize")
tclevrmaterial = baseType("tclevrmaterial")
tclevrshape = baseType("tclevrshape")
tclevrrelation = baseType("tclevrrelation") 
tclevrobject = baseType("tclevrobject")
clevr_types = [
    tclevrcolor, tclevrsize, tclevrmaterial, tclevrshape, tclevrrelation,
    tclevrobject
]

"""Constant Primitives: initialize all the constants as base primitives."""
def clevr_constants():
    colors = [Primitive(f"clevr_{a}", tclevrcolor, a) for a in attribute_constants['color']]
    sizes = [Primitive(f"clevr_{a}", tclevrsize, a) for a in attribute_constants['size']]
    materials = [Primitive(f"clevr_{a}", tclevrmaterial, a) for a in attribute_constants['material']]
    shapes = [Primitive(f"clevr_{a}", tclevrshape, a) for a in attribute_constants['shape']]
    relations = [Primitive(f"clevr_{a}", tclevrrelation, a) for a in attribute_constants['relation']]

    integers = [Primitive(str(j), tint, j) for j in range(10)]
    return colors + sizes + materials + shapes + relations + integers

## Relational handling
def __relate(obj, rel, objset):
    # Takes a list of objects and then returns the subset that is related to that object.
    obj_in_set = [obj1 for obj1 in objset if obj1["id"] == obj["id"]]
    if len(obj_in_set) < 1: return []
    obj_in_set = obj_in_set[0]
    rels = obj_in_set[rel]
    return [obj1 for obj1 in objset if obj1["id"] in rels]
    
def _relate(obj): return lambda rel: lambda objset: __relate(obj, rel, objset)
clevr_relate = Primitive("clevr_relate", 
                        arrow(tclevrobject, tclevrrelation, tlist(tclevrobject), tlist(tclevrobject)), _relate)
    
### Pre-defined filter functions.
def base_filter(condition_fn): # (ObjSet -> ObjSet)
    try:
        return lambda objset: sort_and_dedup_obj_list([obj for obj in objset if condition_fn(obj)])
    except:
        return []

def safe_filter(attribute_type, objset, attr):
    try:
        return sort_and_dedup_obj_list([obj for obj in objset if obj[attribute_type] == attr])
    except:
        return []
def make_filter_handler(attribute_type):
    def filter_handler(objset): # (ObjSet x Attribute -> ObjSet)
        return lambda attr: safe_filter(attribute_type, objset, attr)
    return filter_handler

clevr_filter_color = Primitive("clevr_filter_color", arrow(tlist(tclevrobject), tclevrcolor, tlist(tclevrobject)), make_filter_handler("color"))
clevr_filter_size = Primitive("clevr_filter_size", arrow(tlist(tclevrobject), tclevrsize, tlist(tclevrobject)), make_filter_handler("size"))
clevr_filter_material = Primitive("clevr_filter_material", arrow(tlist(tclevrobject), tclevrmaterial, tlist(tclevrobject)), make_filter_handler("material"))
clevr_filter_shape = Primitive("clevr_filter_shape", arrow(tlist(tclevrobject), tclevrshape, tlist(tclevrobject)), make_filter_handler("shape"))

clevr_filter = Primitive("clevr_filter", arrow(arrow(t0, tbool), tlist(tclevrobject), tlist(tclevrobject)), base_filter)

### Query object attributes and check equality.
def make_query_handler(attribute_type):
    def query_handler(object): 
        return object[attribute_type]
    return query_handler    
def base_attribute_equality(a1):
    return lambda a2: a1 == a2
def _eq_objects(obj1):
    return lambda obj2: obj1['id'] == obj2['id']

clevr_query_color = Primitive("clevr_query_color", arrow(tclevrobject, tclevrcolor), make_query_handler("color"))
clevr_query_size = Primitive("clevr_query_size", arrow(tclevrobject, tclevrsize), make_query_handler("size"))
clevr_query_material = Primitive("clevr_query_material", arrow(tclevrobject, tclevrmaterial), make_query_handler("material"))
clevr_query_shape = Primitive("clevr_query_shape", arrow(tclevrobject, tclevrshape), make_query_handler("shape"))

clevr_eq_color = Primitive("clevr_eq_color", arrow(tclevrcolor, tclevrcolor, tbool), base_attribute_equality)
clevr_eq_size = Primitive("clevr_eq_size", arrow(tclevrsize, tclevrsize, tbool), base_attribute_equality)
clevr_eq_material = Primitive("clevr_eq_material", arrow(tclevrmaterial, tclevrmaterial, tbool), base_attribute_equality)
clevr_eq_shape = Primitive("clevr_eq_shape", arrow(tclevrshape, tclevrshape, tbool), base_attribute_equality)
clevr_eq_objects = Primitive("clevr_eq_objects", arrow(tclevrobject, tclevrobject, tbool), _eq_objects) # Object ID equality.

### Same: filters for objects except for a given object.
def safe_same_handler(attribute_type, obj1, objset):
    # Gets objects that are of the same type as a query object, except for the object itself.
    try: 
        return sort_and_dedup_obj_list([obj2 for obj2 in objset if (obj1[attribute_type] == obj2[attribute_type]) and (obj1["id"] != obj2["id"])])
    except: 
        return []
    
def make_same_handler(attribute_type):
    def same_handler(obj1):  # (Obj -> ObjSet -> Ob jSet)
        return lambda objset: safe_same_handler(attribute_type, obj1, objset)
    return same_handler

def base_filter_except(query_obj):
    try:
        return lambda condition_fn: lambda objset: sort_objs([obj for obj in objset if condition_fn(obj) and obj['id'] != query_obj["id"]])
    except:
        return []
clevr_same_color = Primitive("clevr_same_color", arrow(tclevrobject, tlist(tclevrobject), tlist(tclevrobject)), make_same_handler("color"))
clevr_same_size = Primitive("clevr_same_size", arrow(tclevrobject, tlist(tclevrobject), tlist(tclevrobject)), make_same_handler("size"))
clevr_same_material = Primitive("clevr_same_material", arrow(tclevrobject, tlist(tclevrobject), tlist(tclevrobject)), make_same_handler("material"))
clevr_same_shape = Primitive("clevr_same_shape", arrow(tclevrobject, tlist(tclevrobject), tlist(tclevrobject)), make_same_handler("shape"))

clevr_filter_except = Primitive("clevr_filter_except", arrow(tclevrobject, arrow(t0, tbool), tlist(tclevrobject), tlist(tclevrobject)), base_filter_except)

#### Set operations over the object IDs.
def objset_op(s1, s2, op_fn):
    # Note that we disambiguate between objects by taking from s1
    try:
        s1_ids = set([obj['id'] for obj in s1])
        s2_ids = set([obj['id'] for obj in s2])
        # Hacky:
        if op_fn == 'union':
            result_set = s1_ids | s2_ids 
        elif op_fn == 'intersect':
            result_set = s1_ids & s2_ids
        elif op_fn == 'difference':
            result_set = s1_ids - s2_ids
        else:
            print("Error: unsupported set function.")
        final_ids = set()
        final_set = []
        for o in s1 + s2:
            if o["id"] in result_set and o["id"] not in final_ids:
                final_set.append(o)
                final_ids.add(o["id"])
        return sort_and_dedup_obj_list(final_set)
    except:
        return []

def _set_union(s1): return lambda s2: objset_op(s1, s2, 'union')
def _set_intersect(s1): return lambda s2: objset_op(s1, s2, 'intersect')
def _set_difference(s1) :return lambda s2: objset_op(s1, s2, 'difference')

clevr_union = Primitive("clevr_union", arrow(tlist(tclevrobject), tlist(tclevrobject), tlist(tclevrobject)), _set_union)
clevr_intersect = Primitive("clevr_intersect", arrow(tlist(tclevrobject), tlist(tclevrobject), tlist(tclevrobject)), _set_intersect)
clevr_difference = Primitive("clevr_difference", arrow(tlist(tclevrobject), tlist(tclevrobject), tlist(tclevrobject)), _set_difference)

#### Mathematical operations.
    
clevr_count = Primitive("clevr_count", arrow(tlist(tclevrobject), tint), len)
clevr_eq_int = Primitive("clevr_eq_int", arrow(tint, tint, tbool), _eq)
clevr_is_gt = Primitive("clevr_gt?", arrow(tint, tint, tbool), _gt)
clevr_is_lt = Primitive("clevr_lt?", arrow(tint, tint, tbool), _lt)

# Boolean operations
clevr_not = Primitive("clevr_not", arrow(tbool, tbool), _not)

# Singleton operators
def _safe_car(objset):
    if len(objset) >= 1:
        return objset[0]
    else:
        return NULL_OBJECT

clevr_unique = Primitive("clevr_unique", arrow(tlist(tclevrobject), tclevrobject), _safe_car)
def _exist(objset) : return len(objset) > 0
clevr_exist = Primitive("clevr_exist", arrow(tlist(tclevrobject), tbool), _exist)

# Transformation operators. Returns an object with a given attribute transformed.
def transform_obj(attribute_type, attr, obj):
    return { k : obj[k] if k != attribute_type else attr for k in obj}

def make_transform_handler(attribute_type):
    return lambda attr: lambda o: transform_obj(attribute_type, attr, o)
clevr_transform_color = Primitive("clevr_transform_color", arrow(tclevrcolor, tclevrobject, tclevrobject), make_transform_handler("color"))
clevr_transform_shape = Primitive("clevr_transform_shape", arrow(tclevrshape, tclevrobject, tclevrobject), make_transform_handler("shape"))
clevr_transform_size = Primitive("clevr_transform_size", arrow(tclevrsize, tclevrobject, tclevrobject), make_transform_handler("size"))
clevr_transform_material = Primitive("clevr_transform_material", arrow(tclevrmaterial, tclevrobject, tclevrobject), make_transform_handler("material"))

clevr_test = Primitive("clevr_test", arrow(tclevrcolor, tclevrobject, tclevrobject), make_transform_handler("color"))

### List operators, restricted to be of type object; we can undo this later
def _clevr_safe_map(f,l):
    try:
        return sort_and_dedup_obj_list([f(o) for o in l])
    except:
        return []
def _clevr_map(f): return lambda l: _clevr_safe_map(f,l)

# Removes any duplicate object from the list before adding a new one.
def __clevr_add(obj1, obj_list):
    filtered_list = [o for o in obj_list if o["id"] != obj1["id"]]
    return sort_and_dedup_obj_list([obj1] + filtered_list)
def _clevr_add(o): return lambda l: __clevr_add(o, l)

def safe_tail(obj_list):
    if len(obj_list) == 0: return []
    return obj_list[1:]
    
clevr_map = Primitive("clevr_map", arrow(arrow(tclevrobject, tclevrobject), tlist(tclevrobject), tlist(tclevrobject)), _clevr_map)

def __if(c, t, f):
    return t if c else f
def _clevr_if(c): return lambda t: lambda f: __if(c, t, f)

clevr_if = Primitive("clevr_if", arrow(tbool, t0, t0, t0), _clevr_if)
clevr_is_empty = Primitive("clevr_empty?", arrow(tlist(tclevrobject), tbool), _isEmpty)
clevr_empty = Primitive("clevr_empty", tlist(tclevrobject), [])
clevr_fold = Primitive("clevr_fold", arrow(tlist(tclevrobject), tlist(tclevrobject), arrow(tclevrobject, tlist(tclevrobject), tlist(tclevrobject)), tlist(tclevrobject)), _fold) 
clevr_add = Primitive("clevr_add", arrow(tclevrobject, tlist(tclevrobject), tlist(tclevrobject)), _clevr_add)
clevr_car = Primitive("clevr_car", arrow(tlist(tclevrobject), tclevrobject),  _safe_car)
clevr_cdr = Primitive("clevr_cdr", arrow(tlist(tclevrobject), tlist(tclevrobject)), safe_tail) # Do we need this?


def clevr_map_transform_primitives():
    return [clevr_map] + [clevr_transform_color, clevr_transform_material, clevr_transform_size, clevr_transform_shape]

# N.B. We use 'safe' versions of all functions
def clevr_test_primitives():
    # A minimal test set for development. N.B. we may run into errors with our objects.
    return [clevr_relate] \
    + [clevr_query_color,clevr_query_size, clevr_query_material, clevr_query_shape,
     clevr_eq_color, clevr_eq_size, clevr_eq_material, clevr_eq_shape] \
    + [clevr_union, clevr_intersect] \
    + [clevr_count, clevr_is_gt, clevr_is_lt] \
    + [clevr_not] + [clevr_car, clevr_cdr, clevr_is_empty] \
    + [clevr_if, clevr_empty, clevr_fold, clevr_add] \
    + [clevr_eq_objects] + clevr_constants()  

    
    # Working:
    # [clevr_cdr] + \
    # [clevr_map] + [clevr_transform_color, clevr_transform_material, clevr_transform_size, clevr_transform_shape] + clevr_constants() 
    
def clevr_original_v1_primitives():
    return [clevr_relate] \
    + [clevr_query_color,clevr_query_size, clevr_query_material, clevr_query_shape, clevr_eq_color, clevr_eq_size, clevr_eq_material, clevr_eq_shape]  \
    + [clevr_union, clevr_intersect, clevr_difference] \
    + [clevr_count, clevr_is_gt, clevr_is_lt] \
    + [clevr_unique] \
    + [clevr_filter_material, clevr_filter_size, clevr_filter_color, clevr_filter_shape] \
    + [clevr_same_material, clevr_same_size, clevr_same_color, clevr_same_shape] \
    + clevr_constants() 

def clevr_bootstrap_v1_primitives():
    return [clevr_relate] \
    + [clevr_query_color,clevr_query_size, clevr_query_material, clevr_query_shape,
     clevr_eq_color, clevr_eq_size, clevr_eq_material, clevr_eq_shape] \
    + [clevr_union, clevr_intersect, clevr_difference] \
    + [clevr_count, clevr_is_gt, clevr_is_lt] \
    + [clevr_car] \
    + [clevr_if, clevr_empty, clevr_fold, clevr_add] \
    + clevr_constants()  

def load_clevr_primitives(primitive_names):
    load_all_primitives = primitive_names == ['ALL']
    prims = []
    for pname in primitive_names:
        if pname == 'clevr_test' or load_all_primitives:
            prims += clevr_test_primitives()
        elif pname == 'clevr_original' or load_all_primitives:
            prims += clevr_original_v1_primitives()
        elif pname == 'clevr_bootstrap' or load_all_primitives:
            prims += clevr_bootstrap_v1_primitives()
        elif pname == 'clevr_map_transform' or load_all_primitives:
            prims += clevr_map_transform_primitives()
        elif pname == 'clevr_filter' or load_all_primitives:
            prims += [clevr_filter]
        elif pname == 'clevr_filter_except' or load_all_primitives: 
            prims += [clevr_filter_except]
        elif pname == 'clevr_difference' or load_all_primitives: 
            prims += [clevr_difference]
        else:
            print(f"Unknown primitive set: {pname}")
            assert False
    # Deduplicate the primitive set.
    final_prims = []
    primitive_names = set()
    for primitive in prims:
        if primitive.name not in primitive_names:
            final_prims += [primitive]
            primitive_names.add(primitive.name)
    return final_prims

def generate_ocaml_definitions():
    print("(** CLEVR Types -- types.ml **)")
    for t in clevr_types:
        print(f'let {t.name} = make_ground "{t.name}";;')
        
    print("\n")
    print("(** CLEVR Function Shell Definitions -- program.ml **)")
    primitives = load_clevr_primitives(["ALL"])
    constant_values = []
    for k in attribute_constants:
        constant_values += attribute_constants[k]
        
    for prim in primitives:
        formatted_tp = str.replace(str(prim.tp), '->', '@>')
        if prim.name.split("clevr_")[-1] in constant_values:
            print(f'let primitive_{prim.name} = primitive "{prim.name}" ({formatted_tp}) ("{prim.name.split("clevr_")[-1]}");;')
        else:
            print(f'let primitive_{prim.name} = primitive "{prim.name}" ({formatted_tp}) (TODO);;')

def run_clevr_primitives_test(primitive_names, curriculum):
    # Handle an exemplar of every template type with all instantiations.
    def check_eval(test_task, raw):
        p = Program.parse(raw)
        # Evaluate by hand.
        (x, y) = test_task.examples[0]
        print("IN:")
        for obj in x[0]:
            print(f'ID: {obj["id"]}: Color: {obj["color"]} | Shape: {obj["shape"]} | Mat: {obj["material"]} | Size : {obj["size"]}')
        single_obj = x[0][0]
        print(f"Left: {single_obj['left']}")
        f = p.evaluate([])
        
        def predict(f, x):
            for a in x:
                f = f(a)
            return f
        y_p = predict(f, x)
        print("OUT:")
        for obj in y_p:
            print(f'ID: {obj["id"]}: Color: {obj["color"]} | Shape: {obj["shape"]} | Mat: {obj["material"]} | Size : {obj["size"]}')
        # test_pass = test_task.check(p, timeout=1000)
        # 
        # # print(f"{test_task.name} | pass: {test_pass}")
        # # if not test_pass:
        # #     print(f"Example x: {x}")
        # #     print(f"You said: {y}")
        # #     print(f"Actual: {y_p}")
    
    filter_test = curriculum[0]
    # Test implementing filter by hand.
    
    
    # condition = "(lambda (clevr_eq_size clevr_small (clevr_query_size $0)))"
    condition = "(clevr_eq_size clevr_small (clevr_query_size $1))"
    fold_fn = f"(lambda (lambda (clevr_if {condition} (clevr_add $1 $0) $1)))"
    fold_fn = f"(lambda (lambda (clevr_if {condition} (clevr_add $1 $0) $0)))"
    filter = f"(lambda (clevr_fold $0 clevr_empty {fold_fn}))"
    check_eval(filter_test, filter)
    import pdb; pdb.set_trace()
    
    ## List primitives
    raw = "(lambda (clevr_add (clevr_car $0) $0))"
    check_eval(filter_test, raw)
    
    # raw = "(lambda (clevr_if (clevr_not (clevr_empty? $0)) (clevr_add (clevr_car $0) $0) $0))"
    # check_eval(filter_test, raw)
    # raw =  "(lambda (clevr_add (clevr_car $0) clevr_empty))"
    # check_eval(filter_test, raw)
    
    # Relate
    # raw = "(lambda (clevr_relate (clevr_car $0) clevr_left $0))"
    # check_eval(filter_test, raw)

    # raw = "(lambda (clevr_eq_objects (clevr_car $0) (clevr_car $0)))"
    # check_eval(filter_test, raw)
    
    # CLEVR original
    # raw = "(lambda (clevr_filter_size $0 clevr_large))"
    # check_eval(filter_test, raw)
    # CLEVR with filter
    is_small = "(lambda (clevr_eq_size clevr_small (clevr_query_size $0)))"
    # filter_small = f"(lambda (clevr_filter {is_small} $0))"
    # print(filter_small)
    # check_eval(filter_test, filter_small)
    # 
    # CLEVR same
    # raw = "(lambda (clevr_same_size (clevr_car $0) $0))"
    # check_eval(filter_test, raw)
    
    # filter_small_except = f"(lambda (clevr_filter_except (clevr_car $0) {is_small} $0))"
    # print(filter_small_except)
    # check_eval(filter_test, filter_small_except)
    
    # CLEVR set union
    # raw = "(lambda (clevr_union (clevr_filter_size $0 clevr_small) (clevr_filter_shape $0 clevr_cube)))"
    # check_eval(filter_test, raw)
    
    # Counting 
    # raw = "(lambda (clevr_count (clevr_filter_size $0 clevr_large)))"
    # raw = "(lambda (clevr_gt? (clevr_count (clevr_filter_size $0 clevr_large)) 1)"
    
    # Transformations
    raw = "(lambda (clevr_map (clevr_transform_color clevr_blue) $0))"
    check_eval(filter_test, raw)
    
    
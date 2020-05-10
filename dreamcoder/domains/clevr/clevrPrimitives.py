from dreamcoder.program import Primitive, Program
from dreamcoder.type import *
from dreamcoder.domains.list.listPrimitives import _eq, _map, _fold, _if, _car, _cdr, _cons, _gt, _not, _isEmpty

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
    
#### Constants.
attribute_constants = {
    'color' : ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"],
    'relation' : ["left", "right", "behind", "front"],
    'shape' : ["cube", "sphere", "cylinder"],
    'size' : ["small", "large"],
    'material' : ["rubber", "metal"]
}
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
    obj_in_set = [obj1 for obj1 in obj_set if obj1["id"] == obj["id"]]
    if len(obj_in_set) < 1: return []
    obj_in_set = obj_in_set[0]
    rels = obj_in_set[rel]
    return [obj1 for obj1 in obj_set if obj1["id"] in rels]

def _relate(obj): return lambda rel: lambda objset: __relate(obj, rel, objset)
clevr_relate = Primitive("clevr_relate", 
                        arrow(tclevrobject, tclevrrelation, tlist(tclevrobject)), _relate)

### Get attributes and check equality.
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
clevr_eq_objects = Primitive("clevr_eq_objects", arrow(tclevrobject, tclevrobject, tbool), _eq_objects)

def sort_objs(obj_list):
    return sorted(obj_list, key=lambda o: o["id"])
### Pre-defined filter and pre-defined same
def base_filter(objset): # (ObjSet -> ObjSet)
    return lambda condition_fn: sort_objs([obj for obj in objset if condition_fn(obj)])
def base_filter_except(obj):
    return lambda objset: lambda condition_fn: sort_objs([obj for obj in objset if condition_fn(obj) and obj['id'] != obj]) 
    
def make_filter_handler(attribute_type):
    def filter_handler(objset): # (ObjSet x Attribute -> ObjSet)
        return lambda attr: sort_objs([obj for obj in objset if obj[attribute_type] == attr])
    return filter_handler
def make_same_handler(attribute_type):
    def same_handler(obj1):  # (Obj -> ObjSet -> ObjSet)
        return lambda objset: sort_objs([obj2 for obj2 in objset if obj1[attribute_type] == objs2[attribute_type] and obj1["id"] != obj2["id"]])
    return same_handler
clevr_filter_color = Primitive("clevr_filter_color", arrow(tlist(tclevrobject), tclevrcolor, tlist(tclevrobject)), make_filter_handler("color"))
clevr_filter_size = Primitive("clevr_filter_size", arrow(tlist(tclevrobject), tclevrsize, tlist(tclevrobject)), make_filter_handler("size"))
clevr_filter_material = Primitive("clevr_filter_material", arrow(tlist(tclevrobject), tclevrmaterial, tlist(tclevrobject)), make_filter_handler("material"))
clevr_filter_shape = Primitive("clevr_filter_shape", arrow(tlist(tclevrobject), tclevrshape, tlist(tclevrobject)), make_filter_handler("shape"))

clevr_filter = Primitive("clevr_filter", arrow(tlist(tclevrobject), arrow(t0, tbool), tlist(tclevrobject)), base_filter)

clevr_same_color = Primitive("clevr_same_color", arrow(tclevrobject, tlist(tclevrobject), tlist(tclevrobject)), make_same_handler("color"))
clevr_same_size = Primitive("clevr_same_size", arrow(tclevrobject, tlist(tclevrobject), tlist(tclevrobject)), make_same_handler("color"))
clevr_same_material = Primitive("clevr_same_material", arrow(tclevrobject, tlist(tclevrobject), tlist(tclevrobject)), make_same_handler("color"))
clevr_same_shape = Primitive("clevr_same_shape", arrow(tclevrobject, tlist(tclevrobject), tlist(tclevrobject)), make_same_handler("shape"))

clevr_filter_except = Primitive("clevr_filter_except", arrow(tclevrobject, tlist(tclevrobject), arrow(t0, tbool), tlist(tclevrobject)), base_filter_except)

#### Set operations ovr the object IDs.
def objset_op(s1, s2, op_fn):
    s1_ids = set([obj['id'] for obj in s1])
    s2_ids = set([obj['id'] or obj in s2])
    result_set = op_fn(s1_ids, s2_ids)
    return sort_objs([o for o in s1+s2 if o in result_set])

def _set_union(s1): return lambda s2: objset_op(s1, s2, set.union)
def _set_intersect(s1): return lambda s2: objset_op(s1, s2, set.intersect)
def _set_difference(s1) :return lambda s2: objset_op(s1, s2, set.difference)


clevr_union = Primitive("clevr_union", arrow(tlist(tclevrobject), tlist(tclevrobject)), _set_union)
clevr_intersect = Primitive("clevr_intersect", arrow(tlist(tclevrobject), tlist(tclevrobject)), _set_intersect)
clevr_difference = Primitive("clevr_difference", arrow(tlist(tclevrobject), tlist(tclevrobject)), _set_difference)

#### Mathematical operations.
clevr_count = Primitive("clevr_count", arrow(tlist(tclevrobject), tint), len)
clevr_eq_int = Primitive("clevr_eq_int", arrow(tint, tint, tbool), _eq)
clevr_is_gt = Primitive("clevr_gt?", arrow(tint, tint, tbool), _gt)

# Boolean operations
clevr_not = Primitive("clevr_not", arrow(tbool, tbool), _not)

# Singleton operators
clevr_unique = Primitive("clevr_unique", arrow(tlist(tclevrobject), tclevrobject), _car)
def _exist(objset) : return len(objset) > 0
clevr_exist = Primitive("clevr_exist", arrow(tlist(tclevrobject), tbool), _exist)

# Transformation operators
def make_transform_handler(attribute_type):
    def transform_handler(object): lambda attr: { k : object[k] if k != attribute_type else attr}
    return transform_handler
clevr_transform_color = Primitive("clevr_transform_color", arrow(tclevrobject, tclevrcolor, tclevrobject), make_transform_handler("color"))
clevr_transform_shape = Primitive("clevr_transform_shape", arrow(tclevrobject, tclevrshape, tclevrobject), make_transform_handler("shape"))
clevr_transform_size = Primitive("clevr_transform_size", arrow(tclevrobject, tclevrsize, tclevrobject), make_transform_handler("size"))
clevr_transform_material = Primitive("clevr_transform_material", arrow(tclevrobject, tclevrmaterial, tclevrobject), make_transform_handler("material"))

### List operators, restricted to be of type object; we can undo this later
# TODO: all of these should later sort the list at the end if they return a list.
clevr_if = Primitive("if", arrow(tbool, t0, t0, t0), _if)
clevr_is_empty = Primitive("empty?", arrow(tlist(tclevrobject), tbool), _isEmpty)
clevr_empty = Primitive("clevr_empty", tlist(tclevrobject), [])
clevr_fold = Primitive("clevr_fold", arrow(tlist(t0), t1, arrow(t0, t1, t1), t1), _fold)
clevr_cons = Primitive("clevr_cons", arrow(tclevrobject, tlist(tclevrobject), tlist(tclevrobject)), _cons)
clevr_car = Primitive("clevr_car", arrow(tlist(tclevrobject), tclevrobject), _car)
clevr_cdr = Primitive("clevr_cdr", arrow(tlist(tclevrobject), tlist(tclevrobject)), _cdr)
clevr_map = Primitive("clevr_map", arrow(arrow(tclevrobject, tclevrobject), tlist(tclevrobject), tlist(tclevrobject)), _map)

def clevr_map_transform_primitives():
    return [clevr_map] + [clevr_transform_color, clevr_transform_material, clevr_transform_size, clevr_transform_shape]

def clevr_test_primitives():
    # A minimal test set for development.
    return [clevr_car, clevr_cdr, clevr_cons] + clevr_constants() + [clevr_eq_size, clevr_eq_color, clevr_eq_material, clevr_eq_shape]

def clevr_original_v1_primitives():
    return clevr_constants() 
    + [clevr_relate]
    + [clevr_query_color,clevr_query_size, clevr_query_material, clevr_query_shape,
         clevr_eq_color, clevr_eq_size, clevr_eq_material, clevr_eq_shape] 
    + [clevr_union, clevr_intersect]
    + [clevr_count, clevr_eq_int, clevr_is_gt]
    + [clevr_not]
    # Singleton operators
    + [clevr_unique, clevr_exist]
    # Filter
    + [clevr_filter_material, clevr_filter_size, clevr_filter_color, clevr_filter_shape]
    # Same
    + [clevr_same_material, clevr_same_size, clevr_same_color, clevr_filter_shape]
    # Set difference
    + [clevr_difference]

def clevr_bootstrap_v1_primitives():
    return clevr_constants()  
    + [clevr_relate]
    + [clevr_query_color,clevr_query_size, clevr_query_material, clevr_query_shape,
     clevr_eq_color, clevr_eq_size, clevr_eq_material, clevr_eq_shape] 
    + [clevr_union, clevr_intersect]
    + [clevr_count, clevr_eq_int, clevr_is_gt]
    + [clevr_not]
    # Primitives to build singleton operators
    + [clevr_car, clevr_cdr, clevr_is_empty]
    # Primitives to build filter 
    + [clevr_if, clevr_empty, clevr_fold, clevr_cons]
    # Primitives to build same
    + [clevr_eq_objects]

def load_clevr_primitives(primitive_names):
    prims = []
    for pname in primitive_names:
        if pname == 'clevr_test':
            prims += clevr_test_primitives()
        elif pname == 'clevr_original':
            prims += clevr_original_v1_primitives()
        elif pname == 'clevr_bootstrap':
            prims += clevr_bootstrap_v1_primitives()
        elif pname == 'clevr_map_transform':
            prims += clevr_map_transform_primitives()
        elif pname == 'clevr_filter':
            prims += [clevr_filter]
        elif pname == 'clevr_filter_except' : 
            prims += [clevr_filter_except]
        elif pname == 'clevr_difference': 
            prims += [clevr_difference]
        else:
            print(f"Unknown primitive set: {pname}")
            assert False
    return prims

def generate_ocaml_definitions(primitive_names):
    print("(** CLEVR Types -- types.ml **)")
    for t in clevr_types:
        print(f'let {t.name} = make_ground "{t.name}";;')
        
    print("\n")
    print("(** CLEVR Function Shell Definitions -- program.ml **)")
    primitives = load_clevr_primitives(primitive_names)
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
        single_obj = x[0][0]
        f = p.evaluate([])
        def predict(f, x):
            for a in x:
                f = f(a)
            return f
        y_p = predict(f, x)
        import pdb; pdb.set_trace()
        test_pass = test_task.check(p, timeout=1000)
        
        print(f"{test_task.name} | pass: {test_pass}")
        if not test_pass:
            print(f"Example x: {x}")
            print(f"You said: {y}")
            print(f"Actual: {y_p}")
    
    filter_test = curriculum[0]
    raw = "(lambda (clevr_eq_objects (clevr_car $0) (clevr_car $0)))"
    check_eval(filter_test, raw)
    
    # CLEVR original
    # raw = "(lambda (clevr_filter_size $0 clevr_large))"
    # check_eval(filter_test, raw)
    # CLEVR with filter
    is_large = "(lambda (clevr_query_size $0))"
    check_eval(filter_test, is_large)
    
open Core
open Parser
open Utils
open Type
open Str
open Program
open Re2


(** Note: Cathy wong -- these are defined in program.ml **)
(* RE2 Type Definitions *)
(* let tfullstr = make_ground "tfullstr";;
let tsubstr = make_ground "tsubstr";;

(** Regex constants **)
let primitive_ra = primitive "_a" tsubstr "a";;
let primitive_rb = primitive "_b" tsubstr "b";;
let primitive_rc = primitive "_c" tsubstr "c";;
let primitive_rd = primitive "_d" tsubstr "d";;
let primitive_re = primitive "_e" tsubstr "e";;
let primitive_rf = primitive "_f" tsubstr "f";;
let primitive_rg = primitive "_g" tsubstr "g";;
let primitive_rh = primitive "_h" tsubstr "h";;
let primitive_ri = primitive "_i" tsubstr "i";;
let primitive_rj = primitive "_j" tsubstr "j";;
let primitive_rk = primitive "_k" tsubstr "k";;
let primitive_rl = primitive "_l" tsubstr "l";;
let primitive_rm = primitive "_m" tsubstr "m";;
let primitive_rn = primitive "_n" tsubstr "n";;
let primitive_ro = primitive "_o" tsubstr "o";;
let primitive_rp = primitive "_p" tsubstr "p";;
let primitive_rq = primitive "_q" tsubstr "q";;
let primitive_rr = primitive "_r" tsubstr "r";;
let primitive_rs = primitive "_s" tsubstr "s";;
let primitive_rt = primitive "_t" tsubstr "t";;
let primitive_ru = primitive "_u" tsubstr "u";;
let primitive_rv = primitive "_v" tsubstr "v";;
let primitive_rw = primitive "_w" tsubstr "w";;
let primitive_rx = primitive "_x" tsubstr "x";;
let primitive_ry = primitive "_y" tsubstr "y";;
let primitive_rz = primitive "_z" tsubstr "z";;

let primitive_rvowel = primitive "_rvowel" tsubstr "(a|e|i|o|u)" ;;
let primitive_rconsonant = primitive "_rconsonant" tsubstr "[^aeiou]" ;;
let primitive_emptystr = primitive "_emptystr" tsubstr "";;
let primitive_rdot = primitive "_rdot" tsubstr ".";;
let primitive_rnot = primitive "_rnot" (tsubstr @> tsubstr) (fun s -> "[^" ^ s ^ "]");;
let primitive_ror = primitive "_ror" (tsubstr @> tsubstr @> tsubstr) (fun s1 s2 -> "(("^ s1 ^ ")|("^ s2 ^"))");;
let primitive_rconcat = primitive "_rconcat" (tsubstr @> tsubstr @> tsubstr) (fun s1 s2 -> s1 ^ s2);;

(* RE2 Function Definitions *)
(* Exact regex match *)
let primitive_rmatch = primitive "_rmatch" (tsubstr @> tsubstr @> tboolean) (fun s1 s2 -> 
  try 
    let regex = Re2.create_exn ("^" ^ s1 ^ "$") in
    Re2.matches regex s2 
  with _ -> false
  );;
  
(** Flattens list of substrings back into a string *)
let primitive_rflatten = primitive "_rflatten" (tlist tsubstr @> tfullstr) (fun l -> String.concat ~sep:"" l);;
let primitive_rtail = primitive "_rtail" (tlist tsubstr @> tsubstr) (fun l -> 
  let arr = Array.of_list l in arr.(Array.length arr - 1)
  );;

(** Splits s2 on regex s1 as delimiter, including the matches *)
let not_empty str = (String.length str) > 0;;
let primitive_rsplit = primitive "_rsplit" (tsubstr @> tfullstr @> tlist tsubstr) (fun s1 s2 -> 
  try
    let regex = Re2.create_exn s1 in
    let init_split = Re2.split ~include_matches:true regex s2 in
    (List.filter init_split not_empty)
  with _ -> [s2]
  );;
  
let primitive_rappend = primitive "_rappend" (tsubstr @> tlist tsubstr @> tlist tsubstr) (fun x l -> l @ [x]);;
let primitive_rrevcdr = primitive "_rrevcdr" (tlist tsubstr @> tlist tsubstr) (fun l -> 
  let arr = Array.of_list l in 
  let slice = Array.sub arr 0 (Array.length arr - 1) in
  Array.to_list slice
  );; *)
  

(** Test RE2 Primitives *)
let test_boolean name raw input = 
  let p = parse_program raw |> get_some in
  let p = analyze_lazy_evaluation p in
  let y = run_lazy_analyzed_with_arguments p [input] in 
    Printf.printf "%s | in: %s | out: %s \n" name input (Bool.to_string y);;

let test_single_str name raw input =
  let p = parse_program raw |> get_some in
  let p = analyze_lazy_evaluation p in
  let x = input in
  let y = run_lazy_analyzed_with_arguments p [x] in 
    Printf.printf "%s | out: %s \n" name y
;;

let test_regex name raw input gold =
  let p = parse_program raw |> get_some in
  let p = analyze_lazy_evaluation p in
  let x = input in
  let y = run_lazy_analyzed_with_arguments p [x] in 
    Printf.printf "%s in: %s | out: %s | gold %s \n" name input y gold
;;

let replace_te = "(lambda  (if (_rmatch (_rconcat _t _e) $0) _f $0))";;
let postpend_te = "(lambda (if (_rmatch (_rconcat _t _e) $0) (_rconcat $0 _f) $0))";;
let prepend_te = "(lambda  (if (_rmatch (_rconcat _t _e) $0) (_rconcat _f $0) $0))";;
let remove_te = "(lambda  (if (_rmatch (_rconcat _t _e) $0) _emptystr $0))";;
 
(** Simple matches **)
let simple_test_cases() =
  test_boolean "match ." "(lambda (_rmatch _rdot $0))" "t";
  test_boolean "match ."  "(lambda (_rmatch _rdot $0))" "tt";
  test_boolean "match ab" "(lambda (_rmatch (_rconcat _a _b) $0))" "ab";
  test_boolean "match ab" "(lambda (_rmatch (_rconcat _a _b) $0))" "bab";
  test_boolean "match ab" "(lambda (_rmatch (_rconcat _a _b) $0))" "abc";
  test_boolean "match .bc" "(lambda (_rmatch (_rconcat (_rconcat _rdot _b) _c) $0))" "abc";
  
  (** Or **)
  test_boolean "match b or c" "(lambda (_rmatch (_ror _b _c) $0))" "b";
  test_boolean "match (b|c)d" "(lambda (_rmatch (_rconcat (_ror _b _c) _d) $0))" "bd";
  test_boolean "match (b|c)d" "(lambda (_rmatch (_rconcat (_ror _b _c) _d) $0))" "ed";
  
  (** Not **)
  test_boolean "match [^ae]"  "(lambda (_rmatch (_rnot (_rconcat _a _e)) $0))" "b";
  test_boolean "match [^ae]"  "(lambda (_rmatch (_rnot (_rconcat _a _e)) $0))" "e";
  test_boolean "match [^ae]d"  "(lambda (_rmatch (_rconcat (_rnot (_rconcat _a _e)) _d) $0))" "bd";
  test_boolean "match [^ae]d"  "(lambda (_rmatch (_rconcat (_rnot (_rconcat _a _e)) _d) $0))" "ed";

  (** Single string manipulations *)
  test_single_str "post" postpend_te "te";
  test_single_str "post" postpend_te "z";
  test_single_str "prepend" prepend_te "te";
  test_single_str "replace" replace_te "te";
  
;;
(* simple_test_cases();; *)

let list_test_cases() =
  (** Replace match **)
  let raw = "(lambda (_rflatten (map  "^ replace_te ^ "  (_rsplit (_rconcat _t _e) $0))))" in 
  test_regex "replace te -> f" raw "tehellote"  "fhellof";
  
  (** Prepend to match **)
  let raw = "(lambda (_rflatten (map  "^ prepend_te ^"  (_rsplit (_rconcat _t _e) $0))))" in 
  test_regex "prepend te -> f" raw "tehellote" "ftehellofte";
  
  (** Postpend to match **)
  let raw = "(lambda (_rflatten (map  "^ postpend_te ^"   (_rsplit (_rconcat _t _e) $0) ) ))" in 
  test_regex "postpend te -> f" raw "tehellote" "tefhellotef";
  
  (** Remove match **)
  let raw = "(lambda (_rflatten (map  "^ remove_te ^"   (_rsplit (_rconcat _t _e) $0))))" in 
  test_regex "remove te -> f" raw "teheltelote"  "hello";
  
  (** Match at start **)
  let raw = "(lambda ((lambda (_rflatten (cons ("^ replace_te ^" (car $0)) (cdr $0)))) (_rsplit (_rconcat _t _e) $0)))" in 
  test_regex "replace te -> f" raw "teheltelote" "fheltelote";
  let raw = "(lambda ((lambda (_rflatten (cons ("^ prepend_te ^" (car $0)) (cdr $0)))) (_rsplit (_rconcat _t _e) $0)))" in 
  test_regex "pre te -> f" raw "teheltelote" "fteheltelote";
  let raw = "(lambda ((lambda (_rflatten (cons ("^ postpend_te ^" (car $0)) (cdr $0)))) (_rsplit (_rconcat _t _e) $0)))" in 
  test_regex "post te -> f" raw "teheltelote" "tefheltelote";
  
  (** Match at end **)
  let raw = "(lambda ((lambda (_rflatten (_rappend ("^ replace_te ^ " (_rtail $0)) (_rrevcdr $0)) )) (_rsplit (_rconcat _t _e) $0) ))" in 
  test_regex "replace te -> f" raw "teheltelote" "teheltelof";
  let raw = "(lambda ((lambda (_rflatten (_rappend ("^ postpend_te ^ " (_rtail $0)) (_rrevcdr $0)) )) (_rsplit (_rconcat _t _e) $0) ))" in 
  test_regex "post te -> f" raw "teheltelote" "teheltelotef";
  let raw = "(lambda ((lambda (_rflatten (_rappend ("^ prepend_te ^ " (_rtail $0)) (_rrevcdr $0)) )) (_rsplit (_rconcat _t _e) $0) ))" in 
  test_regex "pre te -> f" raw "teheltelote" "teheltelofte";
  
;;
list_test_cases();;
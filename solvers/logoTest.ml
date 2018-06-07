open LogoLib
open LogoInterpreter

let pi2 = 2. *. 3.1459

let rec unfold f stop b =
if stop b then []
else 
  let x, b' = f b in
   x :: (unfold f stop b')

let star (n : int) : turtle =
  List.fold_left
    (fun (k : turtle) (e : float) : turtle ->
      logo_SEQ
        k
        (logo_SEQ
          (logo_GET
            (fun s ->
              logo_SEQ
                (logo_FW 100.)
                (logo_SET s)
            )
          )
          (logo_RT e)
        )
    )
    logo_NOP
    (List.map
      (fun _ -> pi2 /. (float_of_int n))
      (unfold
        (fun x -> (x, x + 1))
        (fun x -> x >= n)
        0
      )
    )

let _ =
  let c = eval_turtle (star 6) in
  VGWrapper.output_canvas_png c 512 "toto.png" ;
  print_endline "done"

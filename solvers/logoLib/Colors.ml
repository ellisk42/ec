(* h: [0,1] *)
(* s: [0,1] *)
(* l: [0,1] *)

let hsl2rgb h s l =
  if s = 0. then (1.,1.,1.) else
    let h2rgb p q t =
      let t =
        if t < 0. then t+.1. else t
      in
      let t =
        if t > 1. then t-.1. else t
      in
      if t < 1./.6. then p+.(q-.p)*.6.*.t else
      if t < 1./.2. then q else
      if t < 2./.3. then p+.(q-.p)*.(2./.3.-.t)*.6. else
        p
    in

    let q =
      if l < 0.5 then l*.(1.+.s) else l+.s-.l*.s
    in
    let p = 2.*.l -. q in
    (h2rgb p q (h+.1./.3.),
     h2rgb p q h,
     h2rgb p q (h-.1./.3.))

let rgb2hsl r g b =
  let maximum = max r (max b g) in
  let minimum = min r (min b g) in

  if maximum = minimum then (0., 0., maximum) else

    let d = maximum-.minimum in
    let l = (maximum+.minimum)/.2. in
    let s = if l > 0.5 then d/.(2.-.maximum-.minimum) else d/.(maximum+.minimum)
    in

    let h =
      if maximum = r then (g-.b)/.d +. (if g < b then 6. else 0.) else
      if maximum = g then (b-.r)/.d +. 2. else
      if maximum = b then (r-.g)/.d +. 4. else
        assert false
    in
    let h = h/.6. in
    (h,s,l)

let interpolate_color (r1,g1,b1) (r2,g2,b2) =
  let (h1,s1,l1) = rgb2hsl r1 g1 b1 in
  let (h2,s2,l2) = rgb2hsl r2 g2 b2 in
  fun distance -> 
    let h = h1 +. (h2-.h1)*.distance in
    let s = s1 +. (s2-.s1)*.distance in
    let l = l1 +. (l2-.l1)*.distance in
    hsl2rgb h s l


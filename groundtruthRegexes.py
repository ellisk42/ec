
#dict of gt regexes

"""
 		pre.create(".+"),
 		pre.create("\d+"),
        pre.create("\w+"),
        pre.create("\s+"),
        pre.create("\\u+"),
        pre.create("\l+")
"""
gt_dict = {
	
	776:"JPCLN\\u\\u\\u.png",
	922:"WHS\d_\d+",
	354:"(\\u)+",
	523:"(\\u)+|\.",
	184:"\.\d\d\d\d\d\d\d",
	501:"u\d\d",
	760:"\\u\\u",
	49:"(\\u)+\\u\d",
	732:"(\\u|H)R5\d\d",
	450:"-\d(\.(\d)+)?",
	350:"\\u\\u"



}
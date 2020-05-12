#motifs.py

from dreamcoder.program import Program
"""
candidates for filtering:


complex:
- [ ] something on top of bricks



- [ ] filter odd-valued loops?
- [ ] 


todo:
- [ ] test filters
- [X] filter concrete training data
- [X] add filtering to top-level
- [ ] 


"""


#motifs return true if we should filter
brickLayingFn = "#(lambda (lambda (tower_loopM $0 (lambda (lambda (moveHand 3 (reverseHand (tower_loopM $3 (lambda (lambda (moveHand 6 (3x1 $0)))) $0))))))))"

def brickBaseInvention(expr):
	#finds expr where brick laying is followed by any invention
	if not expr.isApplication: return False
	f, args = expr.applicationParse()
	if str(f) == brickLayingFn:
		if any(x.isInvented for _, x in args[-1].walkUncurried()):
			return True 
	return False

def brickBaseReverse(expr):
	#finds expr where brick laying is followed by reverseHand
	if not expr.isApplication: return False
	f, args = expr.applicationParse()
	if str(f) == brickLayingFn:
		if args[-1].isApplication:
			argf, _ = args[-1].applicationParse()
			if str(argf) == 'reverseHand':
				return True 
	return False

def oddLoops(expr):
	# finds expr with loop applied to an odd loop count
	if not expr.isApplication: return False
	f, xs = expr.applicationParse()
	if str(f) == 'tower_loopM':
		if str(xs[0]) in [str(i) for i in [1, 3, 5, 7, 9]]:
			return True
	return False

def oddMoves(expr):
	if not expr.isApplication: return False
	f, xs = expr.applicationParse()
	if str(f) == 'moveHand':
		if str(xs[0]) in [str(i) for i in [1, 3, 5, 7, 9]]:
			return True
	return False


def applyFilter(fn, expr):
	for _, e in expr.walkUncurried():
		if fn(e): return True
	return False

#need testing ... 
def testFilters():

	p = Program.parse("(lambda (#(lambda (lambda (tower_loopM $0 (lambda (lambda (moveHand 3 (reverseHand (tower_loopM $3 (lambda (lambda (moveHand 6 (3x1 $0)))) $0)))))))) 3 6 $0))")

	assert not applyFilter(filterBrickBaseInvention, p)
	assert not applyFilter(filterBrickBaseReverse, p)
	assert not applyFilter(filterOddLoops, p)
	assert not applyFilter(filterOddMoves, p)

	p2 = p.betaNormalForm()

	print(p2)

	assert applyFilter(filterOddMoves, p2)
	assert applyFilter(filterOddLoops, p2)
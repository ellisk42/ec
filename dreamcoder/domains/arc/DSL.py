tdirection = baseType('direction')
tdisplacement = baseType('tdisplacement')
tcolor = baseType('tcolor')
tgrid = baseType('tgrid')
tblock = baseType('tblock')
tbar = baseType('tbar')
tsquare = baseType('tsquare')
tmask = baseType('tmask')
tblocks = tlist(tblock)
tgrids = tlist(tgrid)
tsquares = tlist(tsquares)

def leafPrimitives():

return [

Primitive('1', tint, 1),
Primitive('2', tint, 2),
Primitive('3', tint, 3),
Primitive('4', tint, 4),
Primitive('5', tint, 5),
Primitive('6', tint, 6),
Primitive('7', tint, 7),
Primitive('8', tint, 8),
Primitive('9', tint, 9),

Primitive('true', tbool, True),
Primitive('false', tbool, False),

Primitive("left", tdirection, "left"),
Primitive("right", tdirection, "right"),
Primitive("down", tdirection, "down"),
Primitive("up", tdirection, "up"),
Primitive('diagUL', tdirection, 'diagUL'),
Primitive('diagDL', tdirection, 'diagDL'),
Primitive('diagUR', tdirection, 'diagUR'),
Primitive('diagDR', tdirection, 'diagDR'),


Primitive("black", tcolor, _black),
Primitive("blue", tcolor, _blue),
Primitive("red", tcolor, _red),
Primitive("green", tcolor, _green),
Primitive("yellow", tcolor, _yellow),
Primitive("grey", tcolor, _grey),
Primitive("pink", tcolor, _pink),
Primitive("orange", tcolor, _orange),
Primitive("teal", tcolor, _teal),
Primitive("maroon", tcolor, _maroon)

]

def basePrimitives():

return [

##### tint #####

    Primitive('isOdd', arrow(tint, tbool), lambda x: x%2 == 1, _) # (868de0fa),

##### tblocks #####

    # # arrow (tblocks, tblock)
    Primitive('_head', arrow(tblocks, tblock), _head),
    Primitive('_mergeBlocks', arrow(tblocks, tblock),  _mergeBlocks),
    Primitive('_getListBlock', arrow(tblocks, tint, tblock), _getListBlock),
    # arrow(tblocks, tgrid)
    Primitive('blocksToMinGrid', arrow(tblocks, tgrid), _blocksToMinGrid),
    Primitive('_blocksToGrid',arrow(tblocks, tint, tint, tgrid), _blocksToGrid),
    Primitive('_blocksToGridWithBackground',arrow(tblocks, tint, tint, tcolor, tgrid), _), # (a64e4611)
    Primitive("blocksAsGrid", arrow(tblocks, tgrid, tgrid), _blocksAsGrid),
    Primitive("blocksAsGridColorOverlapping", arrow(tblocks, tgrid, arrow(tcolor, tcolor, tcolor), tgrid), _blocksAsGrid), # coloring function as argument that specifies how to deal with overlapping colors
    Primitive("filterAndMinGrid", arrow(arrow(tblock, tbool), tblocks, tgrid), _filterAndMinGrid),
    # arrow(tblocks, tblocks)
    Primitive('_sortBlocks',arrow(tblocks, arrow(tblock, tint), tblocks), _sortBlocks),
    Primitive("filterBlocks", arrow(arrow(tblock, tbool), tblocks, tblocks), _filter),
    Primitive("mapBlocks", arrow(arrow(tblock, tblock), tblocks, tblocks), _map),
    Primitive("mapAndKeepBlocks", arrow(arrow(tblock, tblock), tblocks, tblocks), _), # (54d82841)
    # arrow(tblocks, tint)
    Primitive('_highestTileBlock', arrow(tblocks, tint), _highestTileBlock),

##### tblock ###### 

    # arrow(tblock, tblock)
    Primitive('applyAndColor', arrow(tblock, arrow(tblock, tblock), tcolor, tblock), _), # change addition to tcolor
    Primitive('fillIn', arrow(tblock, tcolor, tblock), _fillIn),
    Primitive('fill', arrow(tblock, tcolor, tblock), _fill),
    Primitive('reflect', arrow(tblock, tbool, tblock), _reflect),
    Primitive('reflectAboutAxis', arrow(tblock, tbool, tint, tblock), _), # (4c5c2cf0)
    Primitive('move', arrow(tblock, tint, tdirection, tblock), _move),
    Primitive('moveTo', arrow(tblock, tint, tint, tblock), _),
    Primitive('moveUntil', arrow(tblock, arrow(tblock, tbool), tdirection, tblock), _), # (54d82841)
    Primitive('concat', arrow(tblock, tblock, tdirection, tblock), _concat),
    Primitive('concatN', arrow(tblock, tblock, tdirection, tint, tblock), _concatN),
    Primitive('concatUntilEdge', arrow(tblock, tblock, tdirection, tblock), _concatUntilEdge),
    Primitive('duplicate', arrow(tblock, tdirection, tblock), _duplicate),
    Primitive('duplicateN', arrow(tblock, tdirection, tint, tblock), _duplicateN),
    Primitive('duplicateUntilEdge', arrow(tblock, tdirection, tblock), _duplicateUntilEdge),
    Primitive('concatNAndReflect', arrow(tblock, tbool, tdirection, tblock), _concatNAndReflect),
    Primitive('collapseBlock', arrow(tblock, tbool, tblock), _), # tbool controls which axis to collapse on
    Primitive('moveOntoTemplateBlock', arrow(tblock, tblock, tcolor, tcolor, tblock), _), # tcolor to tcolor bijection (97a05b5b)
    Primitive('applyMask', arrow(tblock, tmask, tblock), _), # tblock and tmask must be the exact same size (80af3007)
    Primitive('colorByClosestTile', arrow(tblock, tblock), _),
    Primitive('extendColorUntilEdge', arrow(tblock, tcolor, tdirection, tblock), _), # extend in tdirection if tile not part of block (b8cdaf2b)

    # arrow(tblock, tbool)
    Primitive('hasGeqNTiles', arrow(tblock, tint, tbool), _hasMinTiles),
    Primitive('hasGeqNcolors', arrow(tblock, tint, tbool), _), # (5117e062)
    Primitive('isSymmetrical', arrow(tblock, tbool, tbool), _isSymmetrical),
    Primitive('hasColorNeighbor', arrow(tblock, tcolor, tbool, tbool), _), # if tbool shared corner suffices for neighbor, otherwise shared edge (9edfc990)
    Primitive('isUniqueColor', arrow(tblock, tbool), _), # (0b148d64)
    Primitive('isUniqueShape', arrow(tblock, tbool), _), # (a87f7484)
    Primitive('isColor', arrow(tblock, tcolor, tbool), _),
    Primitive('touchesBoundary', arrow(tblock, tbool), _), # (7b6016b9)
    Primitive('touchesColor', arrow(tblock, tcolor, tbool), _), # (d687bc17)
    # arrow(tblock, trectangle)
    Primitive('forceLargestRectangle', arrow(tblock, trectangle), _), # (98cf29f8)
    # arrow(tblock, tgrid)
    Primitive("blockToGrid", arrow(tblock, tint, tint, tgrid), _blockToGrid),
    Primitive("blockAsGrid", arrow(tblock, tgrid, tgrid), _blockAsGrid),
    # arrow(tblock, tblocks)
    Primitive('split', arrow(tblock, tbool, tblock), _split),
    # arrow(tblock, tmask)
    Primitive('toMask', arrow(trectangle, tmask), _), # (80af3007)
    # arrow(tblock, ttile)
    Primitive('findTilesWithinBlock', arrow(tblock, ttile), _), # (6e19193c)
    # arrow(tblock, tcolor)
    Primitive('findNthColor', arrow(tgrid, tint, tcolor), _findNthColor), # (d687bc17)
    # arrow(tblock, tint)
    Primitive('getNumTiles', arrow(tblock, tint), _), # (c1d99e64)


##### ttile #####

    # arrow(ttile, tblock)
    Primitive('extendSingleDirection', arrow(ttile, tdirection, tint, tblock)),
    Primitive('extendUntilSingleDirection', arrow(ttile, tdirection, arrow(tblock, tbool), tblock), _), # condition given by function (2c608aff)
    Primitive('extendTowardsTile', arrow(ttile, ttile, tblock), _),
    Primitive('extendTowardsTile90', arrow(ttile, ttile, tblock), _), # (a2fd1cf0)
    # arrow(ttile, tdirection)
    Primitive('getTileDirection', arrow(ttile, ttile, tdirection), _), # (dc433765, 4522001f)
    # arrow(ttile, tint)
    Primitive('getTileX', arrow(ttile, tint), _), # (88a10436)
    Primitive('getTileY', arrow(ttile, tint), _), # (88a10436)
    # arrow(ttile, tbool)
    Primitive('HasSharedBlockEdges', arrow(ttile, tint, tblock, tbool), _), #tint: minimum number of shared edges (7e0986d6)

##### ttiles #####

    Primitive('createGrid', arrow(ttiles, tint, tbool, tgrid), _), # tbool: whether to build snakewise, tint: numRows (cdecee7f)

##### tsquare #####

    # arrow(tsquare, tsquare)
    Primitive('fillAlternatingPattern', arrow(tsquare, tcolor, tcolor, tbool, tsquare), _), # if tbool start with first tcolor (b60334d2)
    Primitive('rotate90Clockwise', arrow(tsquare, tsquare), _), # (74dd1130)
    # arrow(tsquare, tbool)
    Primitive('hasEvenSide', arrow(tsquare, tbool), _), # (868de0fa)

##### tbar #####

    Primitive('extendSingleDirection', arrow(tbar, tbool, tint, tbar)),
    Primitive('extendUntilEdgeSingleDirection', arrow(tbar, tbool, tbar)),

##### trectangle #####

    # arrow(trectangle, trectangle)
    Primitive('growOuter', arrow(trectangle, tint, tbool, arrow(tcolor, tcolor), trectangle), _), # tarrow: how to grow, tbool: horizonalSides
    Primitive('shrinkOuter', arrow(trectangle, tint, tbool, arrow(tcolor, tcolor), trectangle), _), # tarrow: how to grow, tbool: horizonalSides
    # arrow(trectangle, tint)
    Primitive('getNumCols', arrow(trectangle, tint), _), # (c1d99e64)
    Primitive('getNumRows', arrow(trectangle, tint), _), # (c1d99e64)
    Primitive('getCenter', arrow(trectangle, ttile), _), # (4c5c2cf0)

##### tcolor ######

    # arrow(tcolor, tcolor)
    Primitive('keepNonBlacks', arrow(tcolor, tcolor, tcolor), _keepNonBlacks),
    Primitive('keepBlackOr', arrow(tcolor, tcolor, tcolor, tcolor), _keepBlackOr),
    Primitive('keepBlackAnd', arrow(tcolor, tcolor, tcolor, tcolor), _keepBlackAnd),
    Primitive('colorIfElse', arrow(tbool, tcolor, tcolor, tcolor), _), # (868de0fa)

##### tgrid #####

    # arrow(tgrid, tblocks)
    Primitive('findSameColorRectangles', arrow(tgrid, tblocks), _findSameColorRectangles),
    Primitive('findSameColorFullRectangles', arrow(tgrid, tblocks), _findSameColorFullRectangles),
    Primitive('findRectanglesBlackB', arrow(tgrid, tblocks), _findRectanglesBlackB),
    Primitive('findRectanglesByB', arrow(tgrid, tcolor, tblocks), _findRectanglesByB),
    
    Primitive('findMinWrappingRectanglesBlackB', arrow(tgrid, tblocks), _), # includes all tiles in the rectangle (80af3007)
    Primitive('findMinWrappingRectanglesByB', arrow(tgrid, tblocks), _), # (80af3007)
    Primitive('findSameColorSquares', arrow(tgrid, tsquares), _), # (80af3007)
    Primitive('findSameColorEblocks', arrow(tgrid, tblocks), _findSameColorEblocks),
    Primitive('findSameColorCblocks', arrow(tgrid, tblocks), _findSameColorCblocks),
    Primitive('findEblocks', arrow(tgrid, tblocks), _findBlocksByEdge),
    Primitive('findCblocks', arrow(tgrid, tblocks), _findBlocksByCorner),
    Primitive('findLooseBlocks', arrow(tgrid, tint, arrow(tgrid, tblocks)), tblocks, _), # adjacent tiles are at most tint away. background (0b148d64, a64e4611)
    Primitive('splitAtBars', arrow(tgrid, tbars, trectangles), _), # (780d0b14)
    # arrow(tgrid, ttiles)
    Primitive('findTilesBlackB', arrow(tgrid, ttiles), _), # (cdecee7f)
    # #arrow(tgrid, tblock)
    Primitive('gridToBlock', arrow(tgrid, tblock), lambda grid: grid),
    # arrow(tgrid, grid)
    Primitive('reflect', arrow(tgrid, tbool, tgrid), _reflect),
    Primitive('grow', arrow(tgrid, tint, tgrid), _grow),
    Primitive('concat', arrow(tgrid, tgrid, tdirection, tgrid), _concat),
    Primitive('concatN', arrow(tgrid, tgrid, tdirection, tint, tgrid), _concatN),
    Primitive('duplicate', arrow(tgrid, tdirection, tgrid), _duplicate),
    Primitive('duplicateN', arrow(tgrid, tdirection, tint, tgrid), _duplicateN),
    Primitive('zipGrids', arrow(tgrid, tgrid, arrow(tcolor, tcolor, tcolor), tgrid), _zipGrids),
    # Primitive('zipGrids2', arrow(tgrids, arrow(tcolor, tcolor, tcolor), tgrid), _zipGrids2),
    Primitive('concatNAndReflect', arrow(tgrid, tbool, tdirection, tgrid), _concatNAndReflect),
    # Primitive('solve0520fde7', arrow(tgrid, tgrid), _solve0520fde7),
    # Primitive('solve007bbfb7', arrow(tgrid, tgrid), _solve007bbfb7),
    # Primitive('solve50cb2852', arrow(tgrid, tcolor, tgrid), _solve50cb2852),
    # Primitive('solvefcb5c309', arrow(tgrid, tgrid), _solvefcb5c309),
    # Primitive('solvec9e6f938', arrow(tgrid, tgrid), _solvec9e6f938),
    # Primitive('solve97999447', arrow(tgrid, tgrid), _solve97999447),
    # Primitive('solvef25fbde4', arrow(tgrid, tgrid), _solvef25fbde4),
    #
    # Primitive('_solve72ca375d', arrow(tgrid, tgrid), _solve72ca375d),
    # Primitive('_solve5521c0d9', arrow(tgrid, tgrid), _solve5521c0d9),
    # Primitive('_solvece4f8723', arrow(tgrid, tgrid), _solvece4f8723),
    # arrow(tgrid, tgridpair)
    Primitive('splitAndMergeGrid', arrow(tgrid, arrow(tcolor, tcolor, tcolor), tbool, tgrid), _splitAndMerge),

#####

##### t0 #####

    # t0
    Primitive("reduce", arrow(arrow(t1, t0, t1), t1, tlist(t0), t1), _reduce),
    Primitive("map", arrow(arrow(t0, t1), tlist(t0), tlist(t1)), _map),
    Primitive("filter", arrow(arrow(t0, tbool), tlist(t0), tlist(t0)), _filter),
    Primitive("mapi",arrow(arrow(tint,t0,t1), tlist(t0), tlist(t1)),_mapi),
    Primitive("filteri", arrow(arrow(tint, t0, t1), tlist(t0), tlist(t1)), _filteri),
    Primitive('length', arrow(tlist(t0), tint), _), # (8efcae92)

##### Task Blueprints #####

    Primitive('findAndMap', arrow(tgrid, arrow(tgrid, tblocks), arrow(tblock, tblock), arrow(tblocks, tgrid), tgrid), _solveGenericBlockMap)

    ]
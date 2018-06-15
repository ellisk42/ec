import Box2D  # The main library
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)

from Box2D.b2 import (
    world,
    polygonShape,
    circleShape,
    staticBody,
    dynamicBody,
    AABB)

import random
import math


class TowerWorld(object):
    def __init__(self):
        self.world = world(gravity=(0, -10), doSleep=True)
        self.ground_body = self.world.CreateStaticBody(
            position=(0, 0),
            shapes=polygonShape(box=(50, 1)),
            userData={"color": (255, 255, 255)}
        )

        self.blocks = []
        self.latestPlan = []

        # self.H = 3.
        # self.W = 0.5
        self.dt = 1. / 10
        self.locationNoise = 0.0

        self.xOffset = 11

    def lowestLegalHeight(self, x, dx, dy):
        lowest = float('-inf')

        x1 = x - dx
        x2 = x + dx

        for b in self.blocks + [self.ground_body]:
            # Fuck you box2d
            assert len(b.fixtures) == 1
            xs = [(b.transform * v)[0]
                  for v in b.fixtures[0].shape.vertices]
            ys = [(b.transform * v)[1]
                  for v in b.fixtures[0].shape.vertices]
            x1_ = min(xs)
            x2_ = max(xs)
            y1_ = min(ys)
            y2_ = max(ys)

            if x1_ > x2 or x1 > x2_:
                continue

            lowest = max(lowest, y2_ + dy)

        return lowest

    def placeBlock(self, x, dx, dy):
        x += self.xOffset

        x += random.random() * self.locationNoise - self.locationNoise / 2

        dx = dx / 2
        dy = dy / 2

        safetyMargin = 0.01
        y = self.lowestLegalHeight(x, dx, dy) + safetyMargin

        body = self.world.CreateDynamicBody(
            position=(
                x,
                y),
            angle=0,
            userData={
                "color": tuple(
                    random.random() *
                    128 +
                    127 for _ in range(3)),
                "p0": (
                    x,
                    y),
                "dimensions": (
                    dx *
                    2,
                    dy *
                    2)})
        box = body.CreatePolygonFixture(box=(dx, dy),
                                        density=1,
                                        friction=1)
        self.blocks.append(body)

    def step(self, dt=None):
        if dt is None:
            dt = self.dt
        self.world.Step(dt, 10, 10)

    def unmoving(self):
        return all(abs(b.linearVelocity[0]) < 0.05 and abs(
            b.linearVelocity[1]) < 0.05 for b in self.blocks)

    def height(self):
        if self.blocks == []:
            return 0
        return max((b.transform * v)[1]
                   for b in self.blocks
                   for v in b.fixtures[0].shape.vertices)

    def length(self):
        if self.blocks == []:
            return 0
        xs = [(b.transform * v)[0]
              for b in self.blocks
              for v in b.fixtures[0].shape.vertices]
        return max(xs) - min(xs)

    def supportedLength(self, height):
        intervals = []
        for b in self.blocks:
            a = b.fixtures[0].body.angle
            if abs(a) > 0.1:
                continue

            ys = [(b.transform * v)[1]
                  for v in b.fixtures[0].shape.vertices]
            if all(y < height for y in ys):
                continue
            xs = [(b.transform * v)[0]
                  for v in b.fixtures[0].shape.vertices]
            x2 = max(xs)
            x1 = min(xs)
            intervals.append((x1, x2))

        def overlap(xxx_todo_changeme, xxx_todo_changeme1):
            (small1, large1) = xxx_todo_changeme
            (small2, large2) = xxx_todo_changeme1
            if large1 < small2:
                return False
            if large2 < small1:
                return False
            return True

        merged = True
        while merged:
            merged = False
            for j, i1 in enumerate(intervals):
                for k, i2 in enumerate(intervals):
                    if k > j and merged == False:
                        if overlap(i1, i2):
                            merged = (j, k)
            if merged:
                j, k = merged
                s1, l1 = intervals[j]
                s2, l2 = intervals[k]
                s = min(s1, s2)
                l = max(l1, l2)
                intervals[j] = (s, l)
                del intervals[k]
        return sum(l - s for s, l in intervals)

    def calculateMask(self, resolution=0.25):
        from scipy.ndimage.morphology import binary_fill_holes
        import numpy as np

        def rounding(z): return int(z / resolution + 0.5)

        h = rounding(self.height()) + 4
        w = rounding(self.length()) + 6
        x0 = min(
            b.userData["p0"][0] -
            b.userData["dimensions"][0] /
            2. for b in self.blocks)

        picture = np.zeros((w, h)).astype(int)

        for b in self.blocks:
            x, y = b.userData["p0"]
            dx, dy = b.userData["dimensions"]

            a = b.fixtures[0].body.angle
            if abs(a) > 0.1:
                continue

            y -= 1  # lower down to the floor
            x -= x0

            dx = rounding(dx / 2.)
            dy = rounding(dy / 2.)
            x = rounding(x) + 1
            y = rounding(y) + 1

            for _dx in range(-dx, dx):
                for _dy in range(-dy, dy):
                    picture[x + _dx, y + _dy] = 1

        # Draw the floor
        picture[:, 0] = 1

        return picture

    def enclosedArea(self):
        from scipy.ndimage.morphology import binary_fill_holes
        import numpy as np

        resolution = 0.25

        picture = self.calculateMask(resolution=resolution)
        flooded = binary_fill_holes(picture).astype(int)
        return resolution * resolution * ((flooded - picture) == 1).sum()

    def overpass(self):
        import numpy as np
        resolution = 0.25
        picture = self.calculateMask(resolution=resolution)
        w,h = picture.shape

        def flood(x,y):
            stack = [(x,y)]
            while len(stack) > 0:
                x,y = stack.pop()
                if x >= 0 and y >= 0 and x < w and y < h and picture[x,y] == 0.:
                    picture[x,y] = 1.
                    stack.append((x - 1,y))
                    stack.append((x + 1,y))
                    stack.append((x,y - 1))
                    stack.append((x,y + 1))

        flood(w - 1, h - 1)
        overpass = 0
        for x in range(w):
            for y in range(h):
                if picture[x,y] == 0.:
                    old = picture.sum()
                    flood(x,y)
                    new = picture.sum()
                    overpass = max(overpass, new - old)
        return overpass*resolution*resolution
        

    def bridge(self):
        import numpy as np

        resolution = 0.25
        originalPicture = self.calculateMask(resolution=resolution)
        # maximum height
        ht = np.where(originalPicture.sum(axis=0) > 0)[0].max()
        # bridge slice
        br = originalPicture[:,ht] > 0

        # 0: have not yet found the bridge
        # 1: walking along the bridge
        # 2: have finished walking along the bridge
        state = 0
        extent = 0
        for b in br:
            if b:
                extent += 1
                if state == 0:
                    state = 1
                elif state == 1:
                    pass
                elif state == 2:
                    return 0.0
            else:
                if state == 0:
                    pass
                elif state == 1:
                    state = 2
                elif state == 2:
                    pass
        return extent*resolution

                




    def staircase(self):
        """Needs to go up and then down"""
        import numpy as np

        resolution = 0.25
        originalPicture = self.calculateMask(resolution=resolution)

        def simulateWalker(picture):
            w, h = picture.shape

            # Are we jumping or falling?
            # 0: have only jumped
            # 1: have also fallen
            state = 0

            # Our simulated staircase Walker
            x = 0
            y = 1

            # Biggest jump / fall sustained
            biggestJump = 0
            biggestFall = 0

            while True:
                if x + 1 >= w:
                    break

                # falling
                fallSize = 0
                while picture[x, y - 1] == 0:
                    fallSize += 1
                    y -= 1
                if fallSize > 0:
                    state = 1
                    
                biggestFall = max(fallSize, biggestFall)

                # Now that we have fallen, see if we can walk forward
                if picture[x + 1, y] == 0:
                    x += 1
                    if x >= w:
                        break

                # jumping
                else:
                    jumpSize = 0
                    while picture[x + 1, y] == 1:
                        y += 1
                        jumpSize += 1
                        if picture[x, y] == 1:
                            # Hit our head on a block
                            biggestJump = float('inf')
                            biggestFall = float('inf')
                            return biggestJump, biggestFall

                        assert y < h,\
                            "Walker hit the ceiling - should be impossible. Plan: %s" % self.originalPlan
                    if jumpSize > 0 and state == 1:
                        # We ended up jumping but were actually only supposed to be falling
                        return float('inf'), float('inf')
                    biggestJump = max(jumpSize, biggestJump)
                    x += 1

            return biggestJump * resolution, biggestFall * resolution

        j1, f1 = simulateWalker(originalPicture)
        j2, f2 = simulateWalker(np.flip(originalPicture, 0))
        j = max(j1, j2)
        f = max(f1, f2)
        return j, f

    def impartImpulses(self, p):
        for b in self.blocks:
            b.ApplyLinearImpulse([random.random() * p - p / 2,
                                  random.random() * p],
                                 b.worldCenter,
                                 True)
            b.ApplyAngularImpulse(random.random() * p - p / 2,
                                  True)

    def stepUntilStable(self):
        for _ in range(25):
            self.step(self.dt)
        # if True: return
        # for _ in range(100000):
        #     self.step(self.dt)
        #     if self.unmoving(): break

    def blocksSignificantlyMoved(self, threshold):
        for b in self.blocks:
            p = (b.worldCenter[0], b.worldCenter[1])
            p0 = b.userData["p0"]
            d = (p[0] - p0[0],
                 p[1] - p0[1])
            r = d[0]**2 + d[1]**2
            if r > threshold:
                return True
        return False

    def executePlan(self, plan):
        self.latestPlan = plan

        for p in plan:
            self.placeBlock(*p)

    def clearWorld(self):
        for b in self.blocks:
            self.world.DestroyBody(b)
        self.blocks = []

    def draw(self, plan):
        import cairo
        import numpy as np

        self.executePlan(plan)

        ppm = 12.  # pixels per meter
        W = 256
        H = 256
        surface = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
        context = cairo.Context(surface)

        for b in [self.ground_body] + self.blocks:
            xs = [(b.transform * v)[0] * ppm
                  for v in b.fixtures[0].shape.vertices]
            ys = [(b.transform * v)[1] * ppm
                  for v in b.fixtures[0].shape.vertices]
            context.set_line_width(1)
            context.move_to(xs[0], ys[0])
            color = [c / 255. for c in b.userData["color"]]
            context.set_source_rgb(*color)
            context.move_to(xs[-1], ys[-1])
            for x, y in zip(xs, ys):
                context.line_to(x, y)
            context.fill()

        a = np.frombuffer(surface.get_data(), np.uint8)
        a.shape = (W, H, 4)
        return np.flip(a[:, :, :3], 0)

    def sampleStability(self, plan, perturbation, N=5):
        hs = []
        wasStable = []
        area = 0
        haveArea = False
        length = 0
        overpass = 0
        biggestJump, biggestFall = None, None
        for _ in range(N):
            self.executePlan(plan)
            initialHeight = self.height()

            if not haveArea:
                area = self.enclosedArea()
                length = self.bridge()
                biggestJump, biggestFall = self.staircase()
                overpass = self.overpass()
                haveArea = True

            hs.append(initialHeight)
            self.impartImpulses(perturbation)
            self.stepUntilStable()
            wasStable.append(self.height() > initialHeight - 0.1 and
                             (not self.blocksSignificantlyMoved(1)))

            # reset the world
            self.clearWorld()
        h = sum(hs) / N
        return {"height": h,
                "stability": sum(wasStable) / float(len(wasStable)),
                "area": area,
                "length": length,
                "overpass": overpass,
                "staircase": max(biggestFall, biggestJump)}


def exportTowers(towers, name):
    from PIL import Image
    import numpy as np

    towers = makeNiceArray(towers)
    m = max(len(t) for t in towers)
    towers = [[TowerWorld().draw(t) for t in ts]
              for ts in towers]

    size = towers[0][0].shape
    tp = towers[0][0].dtype
    towers = [np.concatenate(
        ts + [np.zeros(size, dtype=tp)] * (m - len(ts)), axis=1) for ts in towers]
    towers = np.concatenate(towers, axis=0)
    Image.fromarray(towers).convert('RGB').save(name)


def makeNiceArray(l):
    n = len(l)**0.5
    n = int(n)
    a = []
    while l:
        a.append(l[:n])
        l = l[n:]
    return a

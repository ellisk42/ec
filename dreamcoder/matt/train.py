from dreamcoder.matt.util import *
from dreamcoder.matt.sing import sing
from fastcore.basics import null,ifnone
from tqdm import tqdm
from time import time
import inspect
import numpy as np

def loop_check(locs):
    """
    Q: isnt this inconvenient overkill?
    A: Nothing is overkill if it gets rid of bugs that could mess up
        the validity of results :)
    """
    fail=False
    for k in locs:
        if k not in ('s','t'):
            yellow(f"warning: can't trust local variable {k} in training loop. To indicate that you wont be reusing it between loop steps and dont want to save it, add it to `t`")
            fail=True
    if fail:
        raise ValueError("please resolve local variable warnings by using `s` and `t`")
class FireEvery:
    def __init__(self, j):
        self.j = j
        if self.j is not None and j % sing.cfg.loop.j_multiplier != 0:
            if not sing.cfg.loop.round_j:
                raise ValueError("j_multiplier is misaligned with a loop value. Quick fix with loop.round_j=True")
            # make j a multiple of j_multiplier. Round upwards.
            self.j = sing.cfg.loop.j_multiplier * ((self.j // sing.cfg.j_multiplier) + 1)
    def check(self):
        return self.j is not None and sing.train_state.j % self.j == 0

class RunningValue:
    def __init__(self):
        self.reset()
    def add(self,x):
        self.vals.append(float(x))
    def count(self):
        return len(self.vals)
    def avg(self):
        if self.count() == 0:
            return 0
        return sum(self.vals)/len(self.vals)
    def reset(self):
        self.vals = []
        self.tstart = time()
    def elapsed(self):
        return time() - self.tstart
    def rate(self):
        return self.count() / self.elapsed()

class TrainState:
    def __init__(s):
        s.j = 0
        s.best_valid_loss = np.inf

        s.print_every = FireEvery(sing.cfg.print_every)
        s.valid_every = FireEvery(sing.cfg.valid_every)
        s.search_valid_every = FireEvery(sing.cfg.search_valid_every)
        s.save_every = FireEvery(sing.cfg.save_every)

        s.running_loss = RunningValue()

        if sing.cfg.loop.j_multiplier > sing.cfg.data.buf_size:
            raise ValueError


class Temps(): pass

def main():
    s = sing.train_state

    sing.model.run_tests()

    """
    This is somewhat opinionated, but we want these models to be as portable
    as possible to other frameworks
    therefore I won't actually call .zero_grad .train .eval and with torch.no_grad
    myself but I'll require that it's done in the body of model.train_step and
    model.valid_step.
    """
    with contextlib.suppress(OSError):
        if '.zero_grad()' not in inspect.getsource(sing.model.train_step):
            raise ValueError
        if '.train()' not in inspect.getsource(sing.model.train_step):
            raise ValueError
    with contextlib.suppress(OSError):
        if '.eval()' not in inspect.getsource(sing.model.valid_step):
            raise ValueError
        if '.no_grad()' not in inspect.getsource(sing.model.valid_step):
            raise ValueError

    print(f"Resuming Training at step j={s.j}")

    for s.j in tqdm(range(
                          s.j, # start
                          ifnone(sing.cfg.loop.max_steps,int(1e10)), # stop
                          sing.cfg.j_multiplier) # step
                          ):
        t = Temps()

        # get another batch if needed
        if len(s.frontiers) < s.j_multiplier:
            mlb.red("reloading frontiers")
            s.frontiers += sing.taskloader.getTasks()
            assert len(s.frontiers) > 0

        # pull out the frontiers to train on with this step
        t.fs = [s.frontiers.pop(0) for _ in range(s.j_multiplier)]

        # put back frontiers into cyclic buffer if data.freeze is true
        if sing.cfg.data.train.freeze:
            s.frontiers.extend(fs) # put back at the end
        
        """ 
        Here's what model.train_step should roughly look like:
        model.train_step( whatever data ):
            self.zero_grad()
            loss = call the model on the data
            loss.backward()
            self.optim.step()
            return loss
        """

        t.start = time()
        t.loss, t.to_print = sing.model.train_step(fs)
        t.elapsed = time() - _start
        print(f'Loss {t.loss:.2f} in {t.elapsed:.3f}s on {[f.p for f in t.fs]}')
        if t.to_print is not None: print(t.to_print)

        s.running_loss.add(t.loss)

        mlb.freezer('pause')
        if mlb.predicate('return'):
            return
        if mlb.predicate('which'):
            sing.which()
        if mlb.predicate('cfg'):
            sing.yaml()
        if mlb.predicate('rename'):
            raise NotImplementedError
            name = input('Enter new name:')
            state.rename(name)
            # VERY important to do this:
            w = state.w
            name = state.name

        # printing and logging
        if s.print_every.check():
            print(f"[{s.j}] {s.running_loss.rate():.2g} steps/sec {sing.cls_name} {s.running_loss.avg()}")
            sing.tb_scalar('TrainLoss',s.running_loss.avg())
            s.running_loss.reset()

            sing.model.print_every()


        """ 
        Here's what model.valid_step should roughly look like:
        model.valid_step( whatever data ):
            self.eval()
            loss = call the model on the data
            loss.backward()
            self.optim.step()
            return loss
        """

        # validation loss
        if s.valid_every.check():
            t.valid_loss, t.to_print = sing.model.valid_step(s.valid_frontiers)
            sing.tb_scalar('ValidationLoss',_valid_loss)
            blue(f"[{s.j}] {sing.cls_name} {t.valid_loss}")
            if t.to_print is not None: print(t.to_print)

            # save model if new record for lowest validation loss
            if t.valid_loss < s.best_valid_loss:
                s.best_valid_loss = t.valid_loss
                sing.save('best_validation')
                green('new lowest validation loss!')
                sing.tb_scalar('ValidationLossBest', t.valid_loss)

        # search on validation set
        if s.search_valid_every.check():
            t.model_result = sing.model.search(s.valid_frontiers,
                                              timeout=sing.cfg.loop.search_valid_timeout,
                                              verbose=True)
            sing.tb_scalar('ValidationAccuracy', t.model_result.accuracy())
            raise NotImplementedError #TODO fig out how we wanna to these plots
            sing.tb_plot(t.model_result, file='validation', tb_name=f'ValdiationAccuracy')

        if s.save_every.check():
            loop_check(locals())
            sing.save(f'autosave.{s.j}')


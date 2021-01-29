from dreamcoder.matt.util import *
unthread()
try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.matt import plot,test,train,fix,profile,command,testgen
from dreamcoder.matt.sing import sing

from mlb.mail import email_me,text_me

import sys,os
import numpy as np
import torch
import git
import traceback


# doesnt actually do anything i think
# def tmux_closed():
#     sys.exit(1)
# signal.signal(signal.SIGHUP,tmux_closed)

@hydra.main(config_path="conf", config_name='config')
def hydra_main(cfg):
    np.seterr(all='raise') # so we actually get errors when overflows and zero divisions happen

    if isinstance(cfg.plot.legend,str):
        cfg.plot.legend = eval(cfg.plot.legend)
    if isinstance(cfg.load,str) and cfg.load.strip().startswith('['):
        cfg.load = eval(cfg.load)

    assert sing.cfg.notify_done in ('text','email',None)

    cfg.start_time = str(timestamp())
    cfg.start_time_filename = datetime.datetime.now().strftime(f'%m-%d.%H-%M-%S')
    cfg.argv = ' '.join(sys.argv)
    repo = git.Repo(toplevel_path())
    if repo.is_dirty() and not cfg.dirty:
        raise ValueError("repo is dirty. please add/commit. Or run with `dirty=True`")
    cfg.commit = repo.head.commit.hexsha
    del repo

    def notify_done():
        email_subj = f'[{sing.cfg.mode} done from {sing.cfg.start_time}]'
        email_body = f'{sing.which()}'
        text_body = f'[{sing.cfg.mode} done from {sing.cfg.start_time}]\n{sing.which(no_yaml=True)}'
        if sing.cfg.notify_done == 'email':
            email_me(email_subj,email_body)
        elif sing.cfg.notify_done == 'text':
            text_me(text_body)
            email_me(email_subj,email_body)

    def notify_crash(e:Exception):
        print(sing.which(no_yaml=True))
        full_exc = ''.join(traceback.format_exception(e.__class__, e, e.__traceback__))
        email_subj = f'[{sing.cfg.mode} crash from {sing.cfg.start_time}]'
        email_body = f'{e}\n\n{sing.which()}\n\n{full_exc}'
        text_body = f'[{sing.cfg.mode} crash from {sing.cfg.start_time}]\n{e}\n{sing.which(no_yaml=True)}'
        if sing.cfg.notify_crash == 'email':
            email_me(email_subj,email_body)
        elif sing.cfg.notify_crash == 'text':
            text_me(text_body)
            email_me(email_subj,email_body)


    def ctrlc():
        print(sing.which(no_yaml=True))
         
    with mlb.debug(debug=cfg.debug.mlb_debug, ctrlc=ctrlc, crash=notify_crash):

        sing.from_cfg(cfg) # initialize the singleton

        # PRINT
        if cfg.print:
            print(sing.which()) # as expected: loaded cfg if cfg.load, else new cfg
            print("cfg.print was specified, exiting")
            return

        # PLOT
        elif cfg.mode == 'plot':
            plot.main()
            return

        # TESTGEN
        elif cfg.mode == 'testgen':
            testgen.main()
            return

        # TEST
        elif cfg.mode == 'test':
            mlb.yellow("===START TEST===")
            test.main()
            notify_done()
            mlb.yellow("===TEST DONE===")
            return

        # CMD
        elif cfg.mode == 'cmd':
            command.main()
            return

        # TRAIN
        elif cfg.mode == 'train':
            mlb.yellow("===RESUME TRAIN===")
            print("Entering training loop...")
            train.main()
            notify_done()
            mlb.yellow("===TRAINING DONE===")
            return

        # PROFILE
        elif cfg.mode == 'profile':
            profile.main()
            return

        # INSPECT
        elif cfg.mode == 'inspect':
            print("=== Inspecting State ===")
            s = sing.train_state
            print(sing.which())
            breakpoint()
            raise Exception("Take a look around via `sing` and `s` (train state)!")

        else:
            mlb.die(f"Mode not recognized: {cfg.mode}")


        return

        # PLOT
        if cfg.mode == 'plot':
            if '___' in cfg.load:
                cfg.load = cfg.load.replace('___',' ')
            paths = outputs_regex(*cfg.load.split(' '), sort=cfg.plot.sort)
            # path must at least be DATE/TIME, possibly DATE/TIME/...
            paths = [p for p in paths if len(p.parts) >= 2]

            print("Initial path regex results:")
            for p in paths:
                print(f'\t{p}')

            for i,path in enumerate(paths):
                if 'model_results' not in path.parts:
                    if cfg.plot.suffix is None:
                        mlb.red(f'No plot.suffix provided and the regexed path doesnt contain model_results: {path}')
                        sys.exit(1)
                    paths[i] = get_datetime_path(path) / 'model_results' / cfg.plot.suffix
                    print(f"Path {i} converted to {paths[i]}")

            print("Checking these paths:")
            for p in paths:
                print(f'\t{p}')
                if not p.exists():
                    print(f'\t\t-> DOES NOT EXIST')
            paths = [p for p in paths if p.exists()]
            print("Plotting these paths:")
            for p in paths:
                print(f'\t{p}')
        
            if len(paths) == 0:
                print("No paths to plot")
                sys.exit(1)

            model_results = []
            for path in paths:
                model_results.extend(torch.load(path))
            for m in model_results:
                m.print_dist()
            title = cfg.plot.title if cfg.plot.title is not None else ' '.join(sys.argv[2:])
            if cfg.plot.legend is not None:
                legend = cfg.plot.legend.split('___')
                legend = [x.replace('_',' ') for x in legend]
            else:
                legend = None

            plot.plot_model_results(model_results,
                                    file=cfg.plot.file,
                                    title=title,
                                    toplevel=True,
                                    legend=legend,
                                    tb_name=cfg.plot.tb_name,
                                    cropped=cfg.plot.cropped,
                                    filetype=cfg.plot.filetype,
                                    xlim=cfg.plot.xlim,
                                    save_model_results=False)
            return
        
        # TEST
        elif cfg.mode == 'test':
            original_cfg = None
            tests_from = cfg.test.from_fn or cfg.test.from_file # use fancy python `or` semantics
            if cfg.test.from_fn is not None:
                if cfg.test.from_fn not in test.tests.tests:
                    mlb.red(f"from_fn value not recognized. options are: {list(tests.tests.keys())}")
                    return
                test_frontiers = test.tests.tests[cfg.test.from_fn](cfg)
                mlb.purple(f"got {len(test_frontiers)} test frontiers from {cfg.test.from_fn}()")
                if cfg.test.to_file is not None:
                    print(f"Writing saved tests to {cfg.test.to_file}...")
                    torch.save((test_frontiers,cfg), test.tests.tests_dir / cfg.test.to_file)
                    sys.exit(0)
            elif cfg.test.from_file is not None:
                (test_frontiers,original_cfg) = torch.load(test.tests.tests_dir / cfg.test.from_file)
                # note that original_cfg is just around in case you ever want a record of how the tests were created!
                mlb.green(yaml(original_cfg))
                tests_from = cfg.test.from_file
                test_frontiers = preprocess(test_frontiers,original_cfg)
                mlb.purple(f"loaded {len(test_frontiers)} test frontiers from {cfg.test.from_file} (details in `original_cfg`)")
            else:
                raise ValueError("Specify either test.from_file or test.from_fn")
            assert isinstance(test_frontiers,list) and len(test_frontiers) > 0
            if cfg.load is None:
                print("no state specified to load, exiting")
                return
            ### NOTE: this continues at a later 'test' section
        
        # CMD
        elif cfg.mode == 'cmd':
            cmd.cmd(cfg)
            sys.exit(0)

        # LOAD OR NEW
        print_overrides = []
        state = State()
        if cfg.load is None:
            print("no file to load from, creating new state...")
            if cfg.device != 'cpu':
                with torch.cuda.device(cfg.device):
                    state.new(cfg=cfg)
            else:
                state.new(cfg=cfg)
            # seems to initialize on gpu anyways sometimes
            state.phead = state.phead.to(cfg.device)
            state.vhead = state.vhead.to(cfg.device)
        else:

            paths = outputs_regex(cfg.load)
            # path must at least be DATE/TIME, possibly DATE/TIME/...
            paths = [p for p in paths if len(p.parts) >= 2]
            if len(paths) == 0:
                mlb.red(f'load regex `{cfg.load}` yielded no files:')
            if len(paths) != 1:
                mlb.red(f'load regex `{cfg.load}` yielded multiple possible files:')
                for path in paths:
                    mlb.red(f'\t{path}')
                sys.exit(1)
            [path] = paths

            if 'saves' not in path.parts:
                savefiles = [x for x in (get_datetime_path(path) / 'saves').iterdir()]
                if len([f for f in savefiles if f.name.startswith('autosave.')]) == 0:
                    # old style before we added autosave.j format
                    assert (get_datetime_path(path) / 'saves' / 'autosave').exists()
                    savefile = 'autosave'
                else:
                    savefile = 'autosave.'+str(max([int(x.name.split('.')[1])
                                        for x in (get_datetime_path(path) / 'saves').glob('autosave.*')]))
                path = get_datetime_path(path) / 'saves' / savefile
            
            assert all(['=' in arg for arg in sys.argv[1:]])
            overrides = [arg.split('=')[0] for arg in sys.argv[1:]]

            device = None
            if 'device' in overrides:
                device=cfg.device # override the device

            mlb.green(f"loading from {path}...")
            state.load(path, device=device)

            #mlb.purple(f"It looks like the device of the model is {state.phead.output[0].weight.device}")

            if cfg.mode == 'device':
                mlb.green(f"DEVICE: {state.cfg.device}")
            print("loaded")
            for override in overrides:
                try:
                    # eg override = 'data.T'
                    dotpath = override.split('.')
                    if dotpath[-1] == 'device':
                        raise NotImplementedError
                    target = state.cfg # the old cfg
                    source = cfg # the cfg that contains the overrides
                    for attr in dotpath[:-1]: # all but the last one (which we'll use setattr on)
                        target = target[attr]
                        source = source[attr]
                    overrided_val = source[dotpath[-1]]
                    print_overrides.append(f'overriding {override} to {overrided_val}')
                    with open_dict(target): # disable strict mode
                        target[dotpath[-1]] = overrided_val
                except Exception as e:
                    mlb.red(e)
                    pass
        print()
        print(which(state.cfg))

        for string in print_overrides: # just want this to print after the big wall of yaml
            mlb.purple(string)
            
        with (torch.cuda.device(state.cfg.device) if state.cfg.device != 'cpu' else contextlib.nullcontext()):
            if state.cfg is not None and state.cfg.seed is not None:
                print("Setting evaluation to deterministic (roughly) and seeding RNG")
                torch.manual_seed(state.cfg.seed)
                # warning: these may slow down your model
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                np.random.seed(state.cfg.seed)
                random.seed(state.cfg.seed)

            if cfg.print:
                print(which(state.cfg))
                print("cfg.print was specified, exiting")
                return
            
            mlb.yellow("===START===")


            # TRAIN
            if cfg.mode == 'resume':
                print("Entering training loop...")
                train.train_model(**state.as_kwargs)

            # TEST
            elif cfg.mode == 'test':
                if cfg.test.model_result_path is None:
                    mlb.red("please specify test.model_result_path")
                    sys.exit(1)
                ### NOTE: this continues from the earlier 'test' section
                if original_cfg is not None and original_cfg.test.from_fn == 'deepcoder':
                    test.cfg_diff(state.cfg.data.train,original_cfg.data.test) # print the differences
                if cfg.test.max_tasks is not None and len(test_frontiers) > cfg.test.max_tasks:
                    mlb.red(f"Cutting down test frontiers from {len(test_frontiers)} to {cfg.test.max_tasks}")
                    test_frontiers = test_frontiers[:cfg.test.max_tasks]


                state = fix.fix_state_testmode(state)
                if isinstance(state.phead,SyntaxCheckingRobustFill):
                    state.phead.max_particles = cfg.test.max_particles
                vhead = InvalidIntermediatesValueHead(cfg) if cfg.test.validator_vhead else SampleDummyValueHead()
                solver = make_solver(cfg.data.test.solver,vhead,state.phead,cfg.data.test.max_depth, max_length=cfg.data.test.max_length, max_particles=cfg.test.max_particles, no_resample=cfg.test.no_resample)
                if original_cfg is not None:
                    if state.cfg.data.train.V != original_cfg.data.test.V:
                        mlb.red(mlb.mk_bold(f"HUGE WARNING: You have trained on {state.cfg.data.train.V} data but are testing on {original_cfg.data.test.V}"))
                        exit(1)
                mlb.purple("Running tests")

                if cfg.test.scaffold:
                    mlb.red(mlb.mk_bold("WARNING: SCAFFOLDING IS TURNED ON"))
                    assert hasattr(test_frontiers[0],'scaffold'), "turn off scaffolding or remake ur data"
                    assert test_frontiers[0].scaffold is not None, "make sure expressive lambdas are turned on"


                model_results = test.test_models([solver],
                                            test_frontiers,
                                            state.g,
                                            timeout=cfg.test.timeout,
                                            verbose=True,
                                            scaffold=cfg.test.scaffold,
                                            )
                mlb.purple("plotting results")
                plot.plot_model_results(model_results,
                                        model_result_path=cfg.test.model_result_path,
                                        file=f'{tests_from}_{cfg.test.timeout}s'
                                        )

            # PROFILE
            elif cfg.mode == 'profile':
                mlb.purple('[profiling]')
                import cProfile,pstats
                from pstats import SortKey as sort
                cProfile.runctx('train.train_model(**state.as_kwargs)',globals(),locals(),'profiled')
                p = pstats.Stats('profiled')
                p.strip_dirs()
                p.sort_stats(sort.TIME)
                #p.reverse_order()
                mlb.green('TIME IN FN without children')
                p.sort_stats(sort.TIME)
                p.print_stats(50)
                print('   ncalls  tottime  percall  cumtime  percall filename:lineno(function)')
                print('tottime: doesnt include subfunctions')
                print('percall: previous column divided by num calls')
                mlb.green('CUMULATIVE')
                p.sort_stats(sort.CUMULATIVE)
                p.print_stats(50)
                print('   ncalls  tottime  percall  cumtime  percall filename:lineno(function)')
                print('tottime: doesnt include subfunctions')
                print('percall: previous column divided by num calls')
                breakpoint()

            # INSPECT
            elif cfg.mode == 'inspect':
                print()
                print("=== Inspecting State ===")
                print(which(state.cfg))
                #print(state)
                breakpoint()
                raise Exception("take a look around")
            else:
                raise Exception("Mode not recognized:", cfg.mode)
        # not really sure if this is needed
        #hydra.core.hydra_config.HydraConfig.set_config(cfg)
        mlb.yellow("===END===")
        print(which(state.cfg))

if __name__ == '__main__':
    hydra_main()

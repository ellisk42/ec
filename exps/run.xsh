import sys,os
import argparse
import time

if 'TMUX' in ${...}:
    print("dont run this from within tmux")
    sys.exit(1)

parser = argparse.ArgumentParser()

parser.add_argument('file',
                    type=str,
                    help='experiment file')
parser.add_argument('-f',
                    action='store_true',
                    help='force to kill existing session by same name if it exists')
parser.add_argument('--no-name',
                    action='store_true',
                    help='suppress inserting name=[window name] at the end of the command')
parser.add_argument('--kill',
                    action='store_true',
                    help='only does the -f killing part and doesnt restart it')

args = parser.parse_args()
session = os.path.basename(args.file)

for line in $(tmux ls).split('\n'):
    if not ':' in line:
        continue
    if line.split(':')[0] == session:
        if args.f or args.kill:
            print(f"killing existing tmux session `{session}`")
            tmux kill-session -t @(session)
            pkill -u mlbowers --full prefix=@(session)
        else:
            print(f"tmux session `{session}` exists. Run with -f to force replacing this session")
            sys.exit(1)

if args.kill:
    sys.exit(0)

sessions = {}

for line in open(args.file,'r'):
    line = line.strip()
    if line == '':
        continue
    if ':' not in line:
        print(f"Colon missing in line: {line}, aborting")
        sys.exit(1)
    name, *rest = line.split(':')
    rest = ':'.join(rest) # in case it had any colons in it
    rest = rest.strip()
    if not args.no_name:
        rest = f'cd ~/proj/ec && python bin/test_list_repl.py {rest} prefix={session} name={name}'
    sessions[name] = rest

print(f"launching session `{session}`")
tmux new-session -d -s @(session)

print("launching windows")
for i,(name,cmd) in enumerate(sessions.items()):
    print(f"\t{name}: {cmd}")
    # first make a new window with the right name
    tmux new-window -t @(session) -n @(name)
    # now send keys to the session (which will have the newly created window
    # active already so this will run in that new window
    tmux send-keys -t @(session) @(cmd) C-m
    # C-m is like <CR>
    time.sleep(1.01) # so that the hydra session gets a different name
print("done!")
cmd = f"tmux a -t {session}"
print(f"attach with {cmd}")
tmux a -t @(session)



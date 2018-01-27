import time
import subprocess
import os
import json
import sys


def launch(size = "t2.micro"):
    # aws ec2 run-instances --image-id ami-835f6ae6 --instance-type "t2.micro" --key-name testing --associate-public-ip-address
    o = json.loads(subprocess.check_output(["aws","ec2","run-instances",
                                            "--image-id","ami-835f6ae6",
                                            "--instance-type",size,
                                            "--security-groups","publicssh",
                                            "--instance-initiated-shutdown-behavior","terminate",
                                            "--key-name","testing"]))
    print o
    instance = o[u"Instances"][0][u"InstanceId"]
    print "Launched instance", instance
    
    o = json.loads(subprocess.check_output(["aws","ec2","describe-instances",
                                            "--instance-ids",instance]))
    print o
    address = o[u'Reservations'][0][u'Instances'][0][u'PublicIpAddress']
    print "Retrieved IP address of instance %s; got %s"%(instance,address)
    return instance, address



def sendCommand(address, script):
    import tempfile
    script = """#!/bin/bash
cd ~/ec
git pull
git apply patch
mkdir jobs
""" + script
    fd = tempfile.NamedTemporaryFile(mode = 'w',delete = False,dir = "/tmp")
    fd.write(script)
    fd.close()
    name = fd.name

    # Copy over the script
    print "Copying script over to",address
    os.system("scp -o StrictHostKeyChecking=no -i ~/.ssh/testing.pem %s ubuntu@%s:~/script.sh" % (name, address))

    # delete local copy
    os.system("rm %s"%name)

    # Send git patch
    print "Sending git patch over to",address
    os.system("git diff origin/master | ssh -o StrictHostKeyChecking=no -i ~/.ssh/testing.pem 'cat > ~/ec/patch'" % (name, address))

    # Execute the script
    # For some reason you need to pipe the output to /dev/null in order to get it to detach
    os.system("ssh -o StrictHostKeyChecking=no -i ~/.ssh/testing.pem ubuntu@%s 'bash ./script.sh > /dev/null 2>&1 &'" % (address))
    print "Executing script on remote host."


if __name__ == "__main__":
    instance, address = launch()
    time.sleep(60)
    sendCommand(address, """
    echo starting
    echo test > testOutput
    sleep 100
    echo ending
    """)

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
                                            "--key-name","testing"]))

    instance = o[u"Instances"][0][u"InstanceId"]
    print "Launched instance", instance
    
    o = json.loads(subprocess.check_output(["aws","ec2","describe-instances",
                                            "--instance-ids",instance]))
    address = o[u'Reservations'][0][u'Instances'][0][u'PublicIpAddress']
    print "Retrieved IP address of instance %s; got %s"%(instance,address)
    return instance, address



def sendCommand(address, script):
    script = """
cd ~/ec
git pull
mkdir jobs
%s
"""%script
    command = "ssh -o StrictHostKeyChecking=no -i ~/.ssh/testing.pem ubuntu@%s \"%s\"" % (address, script)
    os.system(command)

instance, address = launch()
time.sleep(120)
sendCommand(address, "echo test > testOutput")

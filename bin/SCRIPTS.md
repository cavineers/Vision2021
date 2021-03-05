## Scripts

start.sh --> Builds new vision instance on nano. Should only be run on completely new Jetsons with JetPack OS installed.

open-update.sh --> Starts Vision instance on nano assuming start has already been run and then update local instance.

open.sh --> Same as open-update.sh but does not pull GitHub repository (should be run when pulling is not possible).

test-86.sh --> Starts already running instance on x86 architectures. (Check Specific Paths and Change as necessary because they will be different per machine).

## To Make Jetson Nano Run open-u.sh on boot use the following
In a new terminal type `sudo nano /etc/rc.local`
Enter this into the nano:

#!/bin/bash

exec > /tmp/rc-local.out 2>&1;set -x

sudo sh /path/to/sh/open.sh
import datetime
import time

while datetime.datetime.now().time() < datetime.time(23, 59, 59, 999999):
    print(1)
    time.sleep(10)

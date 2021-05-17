from datetime import datetime
from datetime import timezone

dt = datetime.now()
dt.replace(tzinfo=timezone.ist)

print(dt.replace(tzinfo=timezone.utc).isoformat())
'2017-01-12 T22:11:31+00:00'

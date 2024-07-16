# workaround for annoying coverage.py bug with path handling
import sqlite3

con = sqlite3.connect(".coverage")
try:
    con.execute(r"UPDATE file SET path = replace(path, '\', '/')")
    con.commit()
finally:
    con.close()

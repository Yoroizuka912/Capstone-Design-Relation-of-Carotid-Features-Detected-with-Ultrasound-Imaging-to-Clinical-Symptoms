import MySQLdb
db = MySQLdb.connect("localhost","root","102936","first" )
cur = db.cursor()
cur.execute()
cur.close()
db.close()
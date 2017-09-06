# coding=utf-8
# need bottle library

import shutil
import sys, time, os
import urllib
import bottle
from bottle import route, template, request
time.clock()


indexhtml = '''
<html lang="zh">
<head>
<title>Named Entity Recognition (Demo Version 1.0)</title>
<meta charset="utf-8">
<link href="//cdn.bootcss.com/bootstrap/4.0.0-alpha.2/css/bootstrap.min.css" rel="stylesheet">
<script src="//cdn.bootcss.com/jquery/2.2.1/jquery.min.js"></script>
<script src="//cdn.bootcss.com/bootstrap/4.0.0-alpha.2/js/bootstrap.min.js"></script>
</head>
<body>
<div class="container">
  <div class="row">
	<hr/>
  	<h1>{{name}}</h1>
  	<h3><small>{{desc}}</small></h3>
	<hr />
  </div>
</div>
<div class="container">
	<div class="row">
		<form class="form" role="form" method="get" action=".">
			<input type="hidden" name="inpage" value="1" />
			<div class="form-group">
				<input type="text" class="form-control" name="p" placeholder="Input your sentence here." value="{{inp if defined('inp') else ''}}"/>
			</div>
			<input type="submit" class="btn btn-primary" value="Submit" />
			% if defined('exams') and len(exams) > 10:
			<button type="button" class="btn" data-toggle="collapse" data-target="#examples">Show Examples</button>
			% end
		</form>
	<div class="row collapse in" id="examples">
		<ul>
			% for item, uitem in exams:
		<li><a href="./?inpage=1&p={{uitem}}">{{item}}</a></li>
			% end
		</ul>
	<hr />
	</div>


	</div>
</div>

<div class="container">
	<div class="row">
	% if defined('inp'):
		Input:
		<h3>{{inp}}</h3>
	% end
	</div>
	<div class="row">
	% if defined('ret'):
		Output:
		<h3>{{!ret}}</h3>
	% end
	</div>
</div>

</body>
</html>
'''

print(sys.argv)
if len(sys.argv) < 2:
	print('USAGE: pyHttpServer.py functionfile [port]')
	sys.exit()

functfile = sys.argv[1]
if os.path.exists('httpmodule.py'):
    os.remove('httpmodule.py')
shutil.copy2(functfile, 'httpmodule.py')
import httpmodule

if len(sys.argv) >= 3:
	port = int(sys.argv[2])
else:
	try:
		port = httpmodule.port
	except:
		print('no port defined!')
		sys.exit()

try:
	name = httpmodule.name
	desc = httpmodule.desc
except:
	name, desc = "LJQ's HTTP Server", ''

try:
	examples = [(x, urllib.parse.quote(x)) for x in httpmodule.examples]
except:
	examples = []

@route('/', method=['GET', 'POST'])
def index():
	p = request.params.p
	inpage = request.params.inpage
	print(p, inpage)
	if p == '': return template(indexhtml, name=name, desc=desc, exams=examples)
	ret = httpmodule.Run(p)
	ret = str(ret)
	if not inpage: return ret
	ret = ret.replace('\n', '<br/>')
	return template(indexhtml, name=name, desc=desc, exams=examples, inp=p, ret=ret)

from bottle import static_file
@route('/static/<filename:path>')
def static(filename):
    return static_file(filename, root='static/')

bottle.run(host='0.0.0.0', port=port)

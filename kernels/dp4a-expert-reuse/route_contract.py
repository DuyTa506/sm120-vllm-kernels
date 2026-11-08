def validate(rows):
 if not 1<=len(rows)<=8:raise ValueError('tokens must be 1..8')
 for row in rows:
  if len(row)!=8:raise ValueError('topk must be8')
  if any(not isinstance(i,int) for i in row):raise ValueError('integer ids required')
  valid=[i for i in row if 0<=i<256]
  if len(set(valid))!=len(valid):raise ValueError('duplicate valid expert within token')

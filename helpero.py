import os
print(os.getcwd())
lines = []
with open('requirements.txt','r') as g:
    for row in g.readlines():
        lines.append(row)
with open('requirements-brand-analysis.txt','r') as f:
    for row in f.readlines():
        print(row[:-1])
        for i in range(len(lines)):
            if lines[i].startswith(row[:-1]):
                print(row[:-1])
                for j in range(len(lines[i])):
                    if lines[i][j]=='=' and lines[i][j+1]=='=':
                        print(lines[i][j+2:])
                        break
                    if lines[i][j]=='@' and lines[i][j+1]==' ':
                        print(lines[i][j+2:])
                        break


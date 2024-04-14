import math

def miscore(predtmp,oritmp):
    score=1
    n=1
    sh=predtmp.shape
    if sh[:-1]!=oritmp.shape[:-1]:
        print('len not match')
        return 1
    
    for i in range(len(sh)-1):
        n=n*sh[i]
    if sh[-1]==3:
        pred=predtmp.reshape(n,3)
    else:
        pred=predtmp.reshape(n,2)
    if oritmp.shape[-1]==3:
        ori=oritmp.reshape(n,3)
    else:
        ori=oritmp.reshape(n,2)
    n=0
    for i in range(len(pred)):
        if ori[i,0]!=0 or ori[i,1]!=0:
            n=n+1
    #print(n)
    tmp=0
    tmpn=0.0
    for i in range(len(pred)):
        if ori[i,0]!=0 or ori[i,1]!=0:
            tmp=abs((math.sqrt(pred[i,0]*pred[i,0]+pred[i,1]*pred[i,1])-math.sqrt(ori[i,0]*ori[i,0]+ori[i,1]*ori[i,1]))/(math.sqrt(pred[i,0]*pred[i,0]+pred[i,1]*pred[i,1])+math.sqrt(ori[i,0]*ori[i,0]+ori[i,1]*ori[i,1])))
            #print(abs(pred[i]-ori[i])/abs(ori[i]))
            score=score-tmp/n
            if tmp<0.2:
                tmpn=tmpn+1
    print('MI(average/percent):')
    print(score,tmpn/n)
    return score,tmpn/n

def siscore(predtmp,oritmp):
    predtmp=predtmp[:,:,:2]
    oritmp=oritmp[:,:,:2]
    score=0
    n=1
    sh=predtmp.shape
    if sh[:-1]!=oritmp.shape[:-1]:
        print('len not match')
        return 1
    for i in range(len(sh)-1):
        n=n*sh[i]
    if sh[-1]==3:
        pred=predtmp.reshape(n,3)
    else:
        pred=predtmp.reshape(n,2)
    if oritmp.shape[-1]==3:
        ori=oritmp.reshape(n,3)
    else:
        ori=oritmp.reshape(n,2)
    n=0
    for i in range(len(pred)):
        if ori[i,0]!=0 and ori[i,1]!=0:
            n=n+1
    #print(n)
    tmp=0
    tmpn=0.0
    for i in range(len(pred)):
        if ori[i,0]!=0 and ori[i,1]!=0:
            tmp=abs((ori[i,0]*pred[i,0]+ori[i,1]*pred[i,1])/math.sqrt(pred[i,0]*pred[i,0]+pred[i,1]*pred[i,1])/math.sqrt(ori[i,0]*ori[i,0]+ori[i,1]*ori[i,1]))
            score=score+tmp/n
            if tmp>0.8:
                tmpn=tmpn+1
            #print(abs(pred[i]-ori[i])/abs(ori[i]))
    print('SI(average/percent):')
    print(score,tmpn/n)

    return score,tmpn/n
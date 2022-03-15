import pandas as pd
import numpy as np
import itertools,json
from sklearn.metrics import mean_squared_error, r2_score,make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans

def process_bar(percent, start_str='', end_str='', total_length=20):
    #Process Bar
    bar = ''.join(['#'] * int(percent * total_length)) + ''
    bar ='\r'+ str(start_str)    + bar.ljust(total_length) + ' {:0>4.1f}%|100% '.format(percent*100) +str(end_str)
    print(bar, end='',flush = True)

def cross_validation(cpm,data_var,time):
    #Output the CV results for each model
    output={}
    for n in data_var:
        if n in [cpm]:
            pass
        else:
            continue
        output2=[]
        varname=data_var[n]['varname']
        data=data_var[n]['data']
        pdata=pd.DataFrame(data[1:time+1],columns=data[0])

        zuhe_base=[]
        zuhe_other=[i for i in varname if i not in zuhe_base]#Uncertain variables
        js=1
        for i2 in range(1,len(zuhe_other)+1):
            varn=tuple(itertools.combinations(zuhe_other,i2))
            
            process_bar(percent=js/(len(zuhe_other)+1), start_str='{} {} '.format(n,i2), end_str='  {}'.format(len(varn)), total_length=20)
            js+=1

            for v in varn:
                zuhe=zuhe_base+list(v)

                X=pdata.loc[:,zuhe].values
                Y=pdata.loc[:,n].values
                Y=Y.reshape(-1,1)
                if len(zuhe)==1:
                    X=X.reshape(-1,1)

                model3 = LinearRegression(fit_intercept = True)  
                model3.fit(X,Y)
                model_cv = LinearRegression(fit_intercept = True)  
                scoreS_v = cross_val_score(model_cv, X, Y, cv=5,scoring=make_scorer(mean_squared_error)) 
                score_v=scoreS_v.mean()

                intercept_all  = model3.intercept_
                coef_all = model3.coef_
                w_all=[i for i in intercept_all]+[i for i in coef_all.flatten()]#Coefficient

                output2.append([list(zuhe),w_all,score_v])
            output2=sorted(output2,key=lambda x : (x[2]) ,reverse=False)[:50000]

        b=sorted(output2,key=lambda x : (x[2]) ,reverse=False)
        output[n]=b[:50000]
    with open(r'data\CVoutput\CVoutput{}_{}.json'.format(cpm,time), 'w') as f:
        json.dump(output,f)

def simple_k_means(X,clusters=100):
    #KMEANS Clustering
    kmeans=KMeans(init='k-means++',n_clusters=clusters)
    kmeans.fit(X)
    labels = kmeans.labels_
    cluster_centers=kmeans.cluster_centers_
    labels_unique = np.unique(labels)


    colors = itertools.cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    out=[]
    for k, col in zip(range(clusters), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        i,z=X[my_members, 0][0],X[my_members, 1][0]
        out.append(i)
    return out

def regularizaiton(tongji,re_range,nx,Y_train,mean_x,std_x,X_test,X_train,Y,CVdeltatime,rank,zuhe,cvscore,X):
    #Regularizaiton with different weights
    for reg in re_range:
        RE1 = Ridge(alpha=reg)
        RE1.fit(X_train,Y_train)
        intercept_RE  = RE1.intercept_
        coef_RE = RE1.coef_
        w_np=np.append(intercept_RE,coef_RE[0],axis=0).reshape(-1,1)
        re_pre=np.dot(np.concatenate((np.ones([len(X_test),1]),X_test),axis=1).astype(float),w_np).flatten().tolist()
        score_re = mean_squared_error(Y[CVdeltatime:],re_pre)
        tongji.append([zuhe,w_np.flatten().tolist(),(score_re+cvscore)/2,score_re,reg,rank])
    return tongji

def regularization_para_change(cpm,Rtime,CVdeltatime,Rdeltatime,data_var,noregul=False):
    #Output the results of regularizaiton
    time=Rtime*Rdeltatime+CVdeltatime

    print('Rtime:',Rtime)

    dz=r'data\CVoutput\CVoutput{}_{}.json'.format(cpm,int(CVdeltatime))
    with open(dz, 'r') as f:
        cv_output=json.load(f)
    if Rtime > 1:         
        dz=r'data\Routput\Routput{}_{}.json'.format(cpm,Rtime-1)
        with open(dz, 'r') as f:
            re_output=json.load(f)
    output={}
    
    for n in cv_output:
        if n in [cpm]:
            pass
        else:
            continue
        if Rtime == 1:
            a1=[0,10000]#Select the top 10,000 CV rankings for RE
            a2=100#Those outside the range are clustered to obtain representative points, and then RE. Reduce running time
            ndata=cv_output[n]
            zhishu_l=[]
            for i in range(a1[1],len(ndata)):
                zhishu=ndata[i][2]
                zhishu_l.append([i,zhishu])
            np_zhishu=np.array(zhishu_l).reshape(-1,2)
            kmeansout=[int(i) for i in simple_k_means(np_zhishu,clusters=a2)]
            re_number=list(range(a1[0],a1[1]))+kmeansout
        else:
            re_number=[]
            rank_reg={}
            ndata=re_output[n]
            for retj in ndata:
                rank=retj[5]
                re_number.append(rank)
                rank_reg[rank]=retj[4]

        data=data_var[n]['data']
        pdata=pd.DataFrame(data[1:time+1],columns=data[0])
        js=1
        sffd=len(re_number)
        tongji1=[]
        for rank in re_number:
            process_bar(percent=js/sffd, start_str='{} {} '.format(n,Rtime,js), end_str=' {}'.format(''), total_length=20)
            js+=1
            zuhe=cv_output[n][rank][0]
            cvscore=cv_output[n][rank][2]
            if Rtime > 1:
                reg_s=round(rank_reg[rank],1)

            X=pdata.loc[:,zuhe].values
            Y=pdata.loc[:,n].values
            Y=Y.reshape(-1,1)

            X_train,X_test=X[:CVdeltatime],X[CVdeltatime:]
            Y_train,Y_test=Y[:CVdeltatime],Y[CVdeltatime:]

            tongji=[]
            re_range=np.arange(0,20,1.01)
            if Rtime > 1:
                if reg_s<=20:
                    re_range=np.arange(0,reg_s+20,1.01)
                else:
                    re_range=np.arange(reg_s-20,reg_s+20,1.01)
                
            if noregul == True:
                re_range=[0]
            tongji=regularizaiton(tongji,re_range,'nx',Y_train,'mean_x','std_x',X_test,X_train,Y,CVdeltatime,rank,zuhe,cvscore,X)
            o=sorted(tongji,key=lambda x : (x[2]) ,reverse=False)[0]
            js12=1
            #Reduced running time
            if noregul != True:
                while o[4]==re_range[-1] or o[4]==re_range[0] :
                    if o[4]<1.5:
                        break
                    if o[4]==re_range[-1]:
                        if o[4]>=200:
                            break
                        re_range=np.arange(o[4]-0.1,o[4]+50,1.01)
                    elif o[4]==re_range[0]:
                        if o[4]<=50:
                            re_range=np.arange(0.0,o[4],1.01)
                        else:
                            re_range=np.arange(o[4]-50.0,o[4]-0.1,1.01)
                    tongji=regularizaiton(tongji,re_range,'nx',Y_train,'mean_x','std_x',X_test,X_train,Y,CVdeltatime,rank,zuhe,cvscore,X)
                    o=sorted(tongji,key=lambda x : (x[2]) ,reverse=False)[0]
                    js12+=1
                    if js12 >=11:
                        break
            #Detailing
            bestreg=o[4]
            if o[4]>1:
                re_range=np.arange(bestreg-1,bestreg+1,0.1)
            else:
                re_range=np.arange(0,bestreg+1,0.1)
            tongji=[]

            tongji=regularizaiton(tongji,re_range,'nx',Y_train,'mean_x','std_x',X_test,X_train,Y,CVdeltatime,rank,zuhe,cvscore,X)
            o=sorted(tongji,key=lambda x : (x[2]) ,reverse=False)[0]

            tongji1.append(o)
        out=sorted(tongji1,key=lambda x : (x[2]) ,reverse=False)
        output[n]=out
        with open(r'data\Routput\Routput{}_{}.json'.format(cpm,Rtime), 'w') as f:
            json.dump(output,f)

def calculatetesterror(cpm,CVdeltatime,Rtime,data_var,model,Rdeltatime):
    #Calculate model error
    w=model[1]
    w_np=np.array(w).reshape(-1,1)
    zuhe=model[0]
    score_cvr=model[2]
    time=Rtime*Rdeltatime+CVdeltatime

    data=data_var[cpm]['data']
    pdata=pd.DataFrame(data[1:time+1],columns=data[0])
    X=pdata.loc[:,zuhe].values
    Y=pdata.loc[:,cpm].values
    Y=Y.reshape(-1,1)

    X_cvr,X_test=X[:time-Rdeltatime],X[time-Rdeltatime:]
    Y_cvr,Y_test=Y[:time-Rdeltatime],Y[time-Rdeltatime:]

    pre=np.dot(np.concatenate((np.ones([len(X_test),1]),X_test),axis=1).astype(float),w_np).flatten().tolist()
    score_test = mean_squared_error(Y_test,pre)
    return score_test**(1/2),score_cvr**(1/2)

def good_models(cpm,models,E,CVdeltatime,Rtime,data_var,Rdeltatime,l):
    #Filter the models within the error threshold
    good_output=[]
    if l==1:
        return good_output
    for m in models[cpm]:
        score_test,score_cvr=calculatetesterror(cpm,CVdeltatime,Rtime,data_var,m,Rdeltatime)
        
        if score_cvr < E:
            if score_test < E:
                a=[m[0],m[1],score_test,score_cvr]
                good_output.append(a)
        else:
            break
    return good_output

def model_judge(cpm,good_m,Rtime):
    #Filtering out models that fit the universal logic
    Excellent_m=[]
    if cpm =='α':
        for m in good_m:
            w=m[1]
            js=0
            for i in w[1:]:
                if i <= -0.3:
                    js+=1
            if js==0:
                Excellent_m.append(m)
    elif cpm =='β':
        for m in good_m:
            w=m[1]
            js=0
            for i in w[1:]:
                if i >= 0.3:
                    js+=1
            if js==0:
                Excellent_m.append(m)
    if Excellent_m!=[]:
        with open(r'data\model_judge\Excellent_m{}_{}.json'.format(cpm,Rtime-1), 'w') as f:
            json.dump(Excellent_m[:3],f)
    return Excellent_m

def Extraction_equation():
    #Generate formula
    eaution_d={'α':[],'β':[]}
    varname_change={'time':'time_i','Increment_Documented_ri':'obs_i','Documented_Infections':'CurPati_i','Recovered':'recovery_i','Cumulative_ri':'Cumucase_i',
                'Increment_Recovered':'INCrec_i','Increment_Deaths':'INCdea_i','Find':'GF_i','Government_ui':'GUI_i','Government_ri':'GDI_i','Government_s':'GS_i',
                'α':'alpha_i','β':'beta_i','INCif_i':'INCif_i','Stay_at_home':'sh_i',
                'Public_Service_and_Public_Place_Closure':'pspc_i',
                'Intra-city_traffic_blockade':'Intracb_i',
                'Inter-city_traffic_blockade':'Intercb_i',
                'Extended_company_holidays':'ech_i',
                'Gathering_activities_are_prohibited':'gap_i',
                'Medical_Resources':'mr_i',
                'Emergency_support_from_doctors_and_nurses':'esdn_i',
                'Emergency_Hospitals':'eh_i',
                'Infection_Search':'is_i',
                'Popularization_of_virus_knowledge':'pvk_i',
                'Administrative_and_legal_support':'als_i',
                'Basic_living_support':'bls_i',
                'Disinfection':'disi_i',
                'Limit_the_number_of_people_gathered':'lnpg_i',
                'Individual_Protection':'inpro_i'}
    models_info={'α':[2,2],'β':[3,1]}
    for variable_name in models_info:
        dz=r'data\model_judge\Excellent_m{}_{}.json'.format(variable_name,models_info[variable_name][0])
        with open(dz, 'r') as f:
            models=json.load(f)
        a=models_info[variable_name][1]
        rezh=models[a][0]
        rew=models[a][1]

        equation=[]
        equation.append('{}*{}'.format(1,rew[0]))
        for v,w in zip(rezh,rew[1:]):
            if '**' in v:
                v_1=v.split('**')
                v_2='state[:,t,{}]**{}'.format(varname_change[v_1[0]],v_1[1])
            elif '*' in v:
                v_1=v.split('*')
                v_2='state[:,t,{}]*state[:,t,{}]'.format(varname_change[v_1[0]],varname_change[v_1[1]])
            else:
                v_2='state[:,t,{}]'.format(varname_change[v])
            equation.append('{}*{}'.format(v_2,w))
        equation='+'.join(equation)
        eaution_d[variable_name].append(equation)


    with open(r'data\equations{}.json'.format(''), 'w') as f:
        json.dump(eaution_d,f)


if __name__ == '__main__':
    #PECR Process
    #Below is the code that has been run. If you want to run the complete code, please remove the comments (#) from lines 331, 340, and 348. 
    #Note: It will take several hours to run the complete code, so make sure you have enough time.
    with open(r'data\data.json', 'r') as f:
        data_var=json.load(f)
    for cpm in ['α','β']:
        print(cpm)
        CVdeltatime=20
        Rdeltatime=10
        E=100
        for l in range(0,6):
            Rtime=l
            if l ==0:
                # cross_validation(cpm,data_var,CVdeltatime)
                dz=r'data\CVoutput\CVoutput{}_{}.json'.format(cpm,int(CVdeltatime))
                with open(dz, 'r') as f:
                    models=json.load(f)
            else:
                model=models[cpm][0]
                Etest,Ecvr=calculatetesterror(cpm,CVdeltatime,Rtime,data_var,model,Rdeltatime)
                print('\n','Etest:',Etest,'Ecvr:',Ecvr,'time:',CVdeltatime+Rtime*Rdeltatime)
                if Ecvr > E:
                    # regularization_para_change(cpm,Rtime,CVdeltatime,Rdeltatime,data_var)
                    dz=r'data\Routput\Routput{}_{}.json'.format(cpm,Rtime)
                    with open(dz, 'r') as f:
                        models=json.load(f)
                else:
                    good_m=good_models(cpm,models,E,CVdeltatime,Rtime,data_var,Rdeltatime,l)
                    Excellent_m=model_judge(cpm,good_m,Rtime)
                    if Excellent_m==[]:
                        # regularization_para_change(cpm,Rtime,CVdeltatime,Rdeltatime,data_var)
                        dz=r'data\Routput\Routput{}_{}.json'.format(cpm,Rtime)
                        with open(dz, 'r') as f:
                            models=json.load(f)
                    else:
                        break 
    Extraction_equation()   
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import matplotlib as mpl
import json

def process_bar(percent, start_str='', end_str='', total_length=20):
    #Process Bar
    bar = ''.join(['#'] * int(percent * total_length)) + ''
    bar ='\r'+ str(start_str)    + bar.ljust(total_length) + ' {:0>5.1f}%|100% '.format(percent*100) +str(end_str)
    print(bar, end='',flush = True)

def NPIs_SEIR_run(runtime,sample_num,equations,AnalyzeStretegy,varname_change,dvpdata,rdata,name):
    #SEIR ordinary differential equation model considering NPIs
    #initial values
    S,N,E,Iu,Id,cumucase=10607700,10607700,960,1060,0,41
    CurPati,death,recovery=38,1,2
    ddt,rdt,rpdt=10,22,7#delay time
    #creat matrixat
    sample_num=sample_num
    runtime=runtime
    state=np.zeros([sample_num,runtime+100,40])
    s_i,e_i,iu_i,id_i,obs_i,CurPati_i,death_i,recovery_i,INCdea_i,INCrec_i,beta_i,alpha_i,GF_i,GS_i,GDI_i,GUI_i,deathrate_i,Cumucase_i,time_i,INCif_i=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
    sh_i,pspc_i,Intracb_i,Intercb_i,ech_i,gap_i,mr_i,esdn_i,eh_i,is_i,pvk_i,als_i,bls_i,disi_i,lnpg_i,inpro_i=20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35

    #add initial values
    state[:,0,[s_i,e_i,iu_i,id_i,CurPati_i,death_i,recovery_i,Cumucase_i]]=np.array([S,E,Iu,Id,CurPati,death,recovery,cumucase])
    #add delay missing values
    incr=[0]+list(round(rdata.loc[:30,'Increment_Recovered']))
    incd=[0]+list(round(rdata.loc[:20,'Increment_Deaths']))
    state[:,6,obs_i]=6
    state[:,:1+rpdt+ddt,INCdea_i]=np.array(incd[:rpdt+1+ddt])
    state[:,:1+rpdt+rdt,INCrec_i]=np.array(incr[:rpdt+1+rdt])
    state[:,rpdt+1:rpdt+1+13,deathrate_i]=np.minimum(np.maximum(np.random.normal(0.25,0.02,np.shape(state[:,rpdt+1:rpdt+1+13,deathrate_i])),0),1)#High mortality rate at the beginning of the epidemic
    state[:,rpdt+1+13:,deathrate_i]=np.minimum(np.maximum(np.random.normal(0.065,0.002,np.shape(state[:,rpdt+1+13:,deathrate_i])),0),1)#Mortality returns to normal after measures
    for t in range(0,runtime):
        process_bar(percent=t/runtime, start_str=name, end_str='{}'.format(t), total_length=20)
        #Improve the simulation effect
        if t>=24 and t<=40:
            et=0.1
        else:
            et=0
        #NPIs
        state[:,t,time_i]=t+1
        npis=['sh_i','pspc_i','Intracb_i','Intercb_i','ech_i','gap_i','mr_i','esdn_i','eh_i','is_i','pvk_i','als_i','bls_i','disi_i','lnpg_i','inpro_i']
        vitem=list(varname_change.items())
        i_var={}
        for vm in vitem:
            i_var[vm[1]]=vm[0]
        for var in AnalyzeStretegy:
            StretegyControl=AnalyzeStretegy[var][1][t]
            forcastingtime=AnalyzeStretegy[var][0]
            if t < forcastingtime:
                if varname_change[var] in npis:
                    npis.remove(varname_change[var])
                for npi in npis:
                    state[:,t,eval(npi)]=dvpdata.loc[t,i_var[npi]]
                state[:,t,eval(varname_change[var])]=dvpdata.loc[t,var]*(1+StretegyControl)
            else:
                if varname_change[var] in npis:
                    npis.remove(varname_change[var])
                for npi in npis:
                    state[:,t,eval(npi)]=state[:,t-1,eval(npi)]
                state[:,t,eval(varname_change[var])]=state[:,t-1,eval(varname_change[var])]*(1+StretegyControl)

        betafit=eval(equations['β'][0])/1000
        alphafit=eval(equations['α'][0])/1000  
        #Error compensation of 0.1
        state[:,t,beta_i]=np.maximum(np.random.normal(betafit-et,0.1,np.shape(state[:,t,beta_i])),0)
        state[:,t,alpha_i]=np.minimum(np.maximum(np.random.normal(alphafit,0.1,np.shape(state[:,t,alpha_i])),0),1)

        # 4th order RungeKutta (RK4) scheme
        ##1
        INCE =state[:,t,s_i]/N * (state[:,t,beta_i]*state[:,t,id_i]+0.55*state[:,t,beta_i]*state[:,t,iu_i])
        INCI=state[:,t,e_i]/4
        INCUI=(1-state[:,t,alpha_i])*state[:,t,e_i]/4
        INCRUI=state[:,t,iu_i]/4
        INCDI=state[:,t,alpha_i]*state[:,t,e_i]/4
        INCRDI=state[:,t,id_i]/4

        INCE,INCI,INCUI,INCRUI,INCDI,INCRDI=np.maximum(INCE,0),         np.maximum(INCI,0),         np.maximum(INCUI,0),         np.maximum(INCRUI,0),         np.maximum(INCDI,0),         np.maximum(INCRDI,0)
        INCE,INCI,INCUI,INCRUI,INCDI,INCRDI=np.random.poisson(INCE),np.random.poisson(INCI),np.random.poisson(INCUI),np.random.poisson(INCRUI),np.random.poisson(INCDI),np.random.poisson(INCRDI)

        STATEinc_S1=-INCE
        STATEinc_E1=INCE-INCI
        STATEinc_UI1=INCUI-INCRUI
        STATEinc_DI1=INCDI-INCRDI
        INCDI1=INCDI

        ##2
        state_S2=state[:,t,s_i] + STATEinc_S1/2
        state_E2=state[:,t,e_i] + STATEinc_E1/2
        state_UI2=state[:,t,iu_i] + STATEinc_UI1/2
        state_DI2=state[:,t,id_i] + STATEinc_DI1/2

        INCE =state_S2/N * (state[:,t,beta_i]*state_DI2+0.55*state[:,t,beta_i]*state_UI2)
        INCI=state_E2/4
        INCUI=(1-state[:,t,alpha_i])*state_E2/4
        INCRUI=state_UI2/4
        INCDI=state[:,t,alpha_i]*state_E2/4
        INCRDI=state_DI2/4

        INCE,INCI,INCUI,INCRUI,INCDI,INCRDI=np.maximum(INCE,0),         np.maximum(INCI,0),         np.maximum(INCUI,0),         np.maximum(INCRUI,0),         np.maximum(INCDI,0),         np.maximum(INCRDI,0)
        INCE,INCI,INCUI,INCRUI,INCDI,INCRDI=np.random.poisson(INCE),np.random.poisson(INCI),np.random.poisson(INCUI),np.random.poisson(INCRUI),np.random.poisson(INCDI),np.random.poisson(INCRDI)

        STATEinc_S2=-INCE
        STATEinc_E2=INCE-INCI
        STATEinc_UI2=INCUI-INCRUI
        STATEinc_DI2=INCDI-INCRDI
        INCDI2=INCDI

        ##3
        state_S3=state[:,t,s_i] + STATEinc_S2/2
        state_E3=state[:,t,e_i] + STATEinc_E2/2
        state_UI3=state[:,t,iu_i] + STATEinc_UI2/2
        state_DI3=state[:,t,id_i] + STATEinc_DI2/2

        INCE =state_S3/N * (state[:,t,beta_i]*state_DI3+0.55*state[:,t,beta_i]*state_UI3)
        INCI=state_E3/4
        INCUI=(1-state[:,t,alpha_i])*state_E3/4
        INCRUI=state_UI3/4
        INCDI=state[:,t,alpha_i]*state_E3/4
        INCRDI=state_DI3/4

        INCE,INCI,INCUI,INCRUI,INCDI,INCRDI=np.maximum(INCE,0),         np.maximum(INCI,0),         np.maximum(INCUI,0),         np.maximum(INCRUI,0),         np.maximum(INCDI,0),         np.maximum(INCRDI,0)
        INCE,INCI,INCUI,INCRUI,INCDI,INCRDI=np.random.poisson(INCE),np.random.poisson(INCI),np.random.poisson(INCUI),np.random.poisson(INCRUI),np.random.poisson(INCDI),np.random.poisson(INCRDI)

        STATEinc_S3=-INCE
        STATEinc_E3=INCE-INCI
        STATEinc_UI3=INCUI-INCRUI
        STATEinc_DI3=INCDI-INCRDI
        INCDI3=INCDI

        ##4
        state_S4=state[:,t,s_i] + STATEinc_S3
        state_E4=state[:,t,e_i] + STATEinc_E3
        state_UI4=state[:,t,iu_i] + STATEinc_UI3
        state_DI4=state[:,t,id_i] + STATEinc_DI3

        INCE =state_S4/N * (state[:,t,beta_i]*state_DI4+0.55*state[:,t,beta_i]*state_UI4)
        INCI=state_E4/4
        INCUI=(1-state[:,t,alpha_i])*state_E4/4
        INCRUI=state_UI4/4
        INCDI=state[:,t,alpha_i]*state_E4/4
        INCRDI=state_DI4/4

        INCE,INCI,INCUI,INCRUI,INCDI,INCRDI=np.maximum(INCE,0),         np.maximum(INCI,0),         np.maximum(INCUI,0),         np.maximum(INCRUI,0),         np.maximum(INCDI,0),         np.maximum(INCRDI,0)
        INCE,INCI,INCUI,INCRUI,INCDI,INCRDI=np.random.poisson(INCE),np.random.poisson(INCI),np.random.poisson(INCUI),np.random.poisson(INCRUI),np.random.poisson(INCDI),np.random.poisson(INCRDI)

        STATEinc_S4=-INCE
        STATEinc_E4=INCE-INCI
        STATEinc_UI4=INCUI-INCRUI
        STATEinc_DI4=INCDI-INCRDI
        INCDI4=INCDI

        ##state update
        state[:,t+1,s_i]=state[:,t,s_i] - np.round(STATEinc_S1/6+STATEinc_S2/3+STATEinc_S3/3+STATEinc_S4/6)
        state[:,t+1,e_i]=state[:,t,e_i] + np.round(STATEinc_E1/6+STATEinc_E2/3+STATEinc_E3/3+STATEinc_E4/6)
        state[:,t+1,iu_i]=state[:,t,iu_i] + np.round(STATEinc_UI1/6+STATEinc_UI2/3+STATEinc_UI3/3+STATEinc_UI4/6)
        state[:,t+1,id_i]=state[:,t,id_i] + np.round(STATEinc_DI1/6+STATEinc_DI2/3+STATEinc_DI3/3+STATEinc_DI4/6)
        
        state[:,t+1,INCif_i]=np.round(INCDI1/6+INCDI2/3+INCDI3/3+INCDI4/6)
        state[:,t+1+rpdt,obs_i]=np.round(INCDI1/6+INCDI2/3+INCDI3/3+INCDI4/6)

        for dit in range(sample_num):
            dldeath=np.round(np.random.gamma(ddt,1,int(state[dit,t+1+rpdt,obs_i]*state[dit,t+1+rpdt,deathrate_i])))
            for i in dldeath:
                state[dit,t+1+rpdt+int(i),INCdea_i]=state[dit,t+1+rpdt+int(i),INCdea_i]+1
        for dit in range(sample_num):
            dlrec=np.round(np.random.gamma(rdt,1,int(state[dit,t+1+rpdt,obs_i]*(1-state[dit,t+1+rpdt,deathrate_i]))))
            for i in dlrec:
                state[dit,t+1+rpdt+int(i),INCrec_i]=state[dit,t+1+rpdt+int(i),INCrec_i]+1

        state[:,t+1,death_i]=state[:,t,death_i]+state[:,t+1,INCdea_i]
        state[:,t+1,recovery_i]=state[:,t,recovery_i]+state[:,t+1,INCrec_i]
        state[:,t+1,CurPati_i]=state[:,t,CurPati_i] + state[:,t+1,obs_i] - state[:,t+1,INCdea_i] - state[:,t+1,INCrec_i]
        state[:,t+1,Cumucase_i]=state[:,t+1,CurPati_i]+state[:,t+1,recovery_i]+state[:,t+1,death_i]
    print('')
    return state

def ouputplot(state_baseline,state_up,state_down,varname_change,rdata):
    #Output the graphical results of the simulation
    plt.rcParams['font.sans-serif']=['Times New Roman']
    js=1
    official_name={
        'Increment_Documented_ri':'New Confirmed Cases','Increment_Deaths':'Simulation Result of New Deaths',
        'Increment_Recovered':'Simulation Result of New Recovered Cases','Documented_Infections':'Existing Confirmed Cases',
        'Intra-city_traffic_blockade':'Intra-city_traffic_blockade','Cumulative_ri':'Cumulative_cases',
        'α':r'α','β':r'β'}
    s_i,e_i,iu_i,id_i,obs_i,CurPati_i,death_i,recovery_i,INCdea_i,INCrec_i,beta_i,alpha_i,GF_i,GS_i,GDI_i,GUI_i,deathrate_i,Cumucase_i,time_i,INCif_i=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
    sh_i,pspc_i,Intracb_i,Intercb_i,ech_i,gap_i,mr_i,esdn_i,eh_i,is_i,pvk_i,als_i,bls_i,disi_i,lnpg_i,inpro_i=20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35
    for i in ['α','β','Documented_Infections','Increment_Documented_ri','Increment_Deaths','Increment_Recovered','Cumulative_ri']:
        if i in ['α','β']:
            a=1000
            yname=i
            p1=plt.figure()
            ax = plt.subplot(1,1,1)
            js+=1
            plt.sca(ax)
            ax.set_title('{}'.format(official_name[i]), color='k')
        else:
            a=1
            yname='Cases'
            p1=plt.figure()
            ax = plt.subplot(1,1,1)
            js+=1
            plt.sca(ax)
            ax.set_title('{}'.format(official_name[i]), color='k')
        for state,strategy_name,color_ma in zip([state_baseline,state_up,state_down],['Baseline','Up','Down'],['#1f77b4','#ff7f0e','#2ca02c']):
            if i in AnalyzeStretegy_baseline:
                ax.plot(range(1,runtime),np.mean(state[:,:runtime-1,eval(varname_change[i])],0),label='Mean: {}'.format(strategy_name),linewidth=2,c=color_ma)
            else:
                ax.plot(range(1,runtime),np.mean(state[:,1:runtime,eval(varname_change[i])],0),label='Mean: {}'.format(strategy_name),linewidth=2,c=color_ma)
            ax.fill_between(range(1,runtime),np.min(state[:,1:runtime,eval(varname_change[i])],0),np.max(state[:,1:runtime,eval(varname_change[i])],0),alpha=0.3,label='Range: {}'.format(strategy_name),facecolor=color_ma)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(int(runtime/10)))
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        plt.xlabel(u'Time')
        plt.ylabel(u'{}'.format(yname))
        plt.legend()
        plt.tick_params()
        plt.show()

if __name__ == '__main__':
    #Main Code
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
    AnalyzeStretegy_baseline={
        'Intra-city_traffic_blockade':                [60,[0]*60 + [-0.0]*30],#alpha & beta
        'Gathering_activities_are_prohibited':        [60,[0]*60 + [-0.0]*30],#beta
        'Emergency_Hospitals':                        [60,[0]*60 + [-0.0]*30],#alpha & beta
        'Disinfection':                               [60,[0]*60 + [-0.0]*30],#beta
        }
    AnalyzeStretegy_up={
        'Intra-city_traffic_blockade':                [60,[0.5]*60 + [-0.0]*30],#alpha & beta
        'Gathering_activities_are_prohibited':        [60,[0]*60 + [-0.0]*30],#beta
        'Emergency_Hospitals':                        [60,[0]*60 + [-0.0]*30],#alpha & beta
        'Disinfection':                               [60,[0]*60 + [-0.0]*30],#beta
        }
    AnalyzeStretegy_down={
        'Intra-city_traffic_blockade':                [60,[-0.5]*60 + [-0.0]*30],#alpha & beta
        'Gathering_activities_are_prohibited':        [60,[-0]*60 + [-0.0]*30],#beta
        'Emergency_Hospitals':                        [60,[-0]*60 + [-0.0]*30],#alpha & beta
        'Disinfection':                               [60,[-0]*60 + [-0.0]*30],#beta
        }
    sample_num=100
    runtime=60

    #load
    with open(r'data\equations.json', 'r') as f:
        equations=json.load(f)#dict
    with open(r'data\data.json', 'r') as f:
        data_var=json.load(f)#dict
    dvpdata=pd.DataFrame(data_var['β']['data'][1:],columns=data_var['β']['data'][0])
    with open(r'data/truedata.json'.format(''), 'r') as f:
        realdata=json.load(f)#dict
    rdata=pd.DataFrame(realdata[1:],columns=realdata[0])

    state_baseline=NPIs_SEIR_run(runtime,sample_num,equations,AnalyzeStretegy_baseline,varname_change,dvpdata,rdata,'baseline')
    state_up=NPIs_SEIR_run(runtime,sample_num,equations,AnalyzeStretegy_up,varname_change,dvpdata,rdata,'up')
    state_down=NPIs_SEIR_run(runtime,sample_num,equations,AnalyzeStretegy_down,varname_change,dvpdata,rdata,'down')

    ouputplot(state_baseline,state_up,state_down,varname_change,rdata)
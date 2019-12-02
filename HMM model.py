from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
from collections import  defaultdict
import pickle
import pandas as pd
import numpy as np
import operator
import math
import collections
import pandas as pd
from nltk.tokenize import wordpunct_tokenize
import numpy as np
import json
import re



exclude=['</s>','<s>','<vars>','regexxxxvar']

dictionaryMatch=defaultdict(list)
dictionaryMatch['LiteralString'].append('literals') 
dictionaryMatch['<vars>'].append('literals') 
dictionaryMatch['VariableDeclaration'].append('var') 
dictionaryMatch['var'].append('var') 
dictionaryMatch['let'].append('var') 
dictionaryMatch['IfStatement'].append('if') 
dictionaryMatch['if'].append('if') 
dictionaryMatch['ReturnStatement'].append('return') 
dictionaryMatch['return'].append('return') 
dictionaryMatch['NewExpression'].append('new') 
dictionaryMatch['new'].append('new') 
dictionaryMatch['ThisExpression'].append('this') 
dictionaryMatch['this'].append('this') 
dictionaryMatch['FunctionExpression'].append('fxn') 
dictionaryMatch['function'].append('fxn') 
dictionaryMatch['ThrowStatement'].append('throw') 
dictionaryMatch['throw'].append('throw') 
dictionaryMatch['FunctionDeclaration'].append('fxn') 
dictionaryMatch['LiteralBoolean'].append('bool') 
dictionaryMatch['true'].append('bool') 
dictionaryMatch['false'].append('bool') 
dictionaryMatch['LiteralNull'].append('null') 
dictionaryMatch['null'].append('null') 
dictionaryMatch['BreakStatement'].append('br') 
dictionaryMatch['break'].append('br') 
dictionaryMatch['SwitchCase'].append('sw') 
dictionaryMatch['switch'].append('sw') 
dictionaryMatch['SwitchStatement'].append('sw') 
dictionaryMatch['CatchClause'].append('cat') 
dictionaryMatch['catch'].append('cat')
dictionaryMatch['EmptyStatement'].append('es') 
dictionaryMatch['empty'].append('es')
dictionaryMatch['ForStatement'].append('for') 
dictionaryMatch['ForInStatement'].append('for')
dictionaryMatch['for'].append('for')
dictionaryMatch['DoWhileStatement'].append('do')
dictionaryMatch['do'].append('do')
dictionaryMatch['WhileStatement'].append('while')
dictionaryMatch['while'].append('while')
dictionaryMatch['TryStatement'].append('try') 
dictionaryMatch['try'].append('try')


dictionarymatch2=dict((k.lower(), v[0].lower()) for k,v in dictionaryMatch.items())



def getJsonInfo(data):        
    dict2 = {}
    for l2 in data:
         if(l2 != 0 and l2['id']!=0):
           if (('ConditionalExpression' not in l2['type']) and ('AssignmentExpression' not in l2['type']) and ('SequenceExpression' not in l2['type']) and ('LogicalExpression' not in l2['type']) and ('CallExpression' not in l2['type']) and ('BinaryExpression' not in l2['type']) and ('UnaryExpression' not in l2['type']) and ('MemberExpression' not in l2['type']) and ('ArrayExpression' not in l2['type']) and ('Block' not in l2['type'])and ('ExpressionStatement' not in l2['type'])and ('AssignmentExpression' not in l2['type']) and ('ObjectExpression' not in l2['type']) and ('ArrayAccess' not in l2['type'])and ('UpdateExpression' not in l2['type'])):
            k = l2['id']
            dict2[k] = l2
    return dict2


def wordtdagconvertor(new_dict,ts):
    #print("div",new_dict)
    totalsentence=ts
    totalleng=len(totalsentence)
    #print(totalsentence)
    wordtag2 = defaultdict(list)
    i=0         
    for keys in new_dict.keys():
        k=0
        flag=0
        while (k<5):
            k+=1
            if new_dict[keys]['type']=='VariableDeclarator' or new_dict[keys]['type']=='LiteralRegExp' or new_dict[keys]['type']=='LabeledStatement' or new_dict[keys]['type']=='LiteralNumber' or new_dict[keys]['type']=='Property'  or new_dict[keys]['type']=='Identifier':
                temp=i+k
                #print(temp)
                if  'value' not in new_dict[keys].keys():
                    continue
                if temp>totalleng-1:
                    break
                if str(new_dict[keys]['value']).isdigit()==False and (new_dict[keys]['value'].lower()) == (totalsentence[temp]):
                    #print('match')
                    strr=new_dict[keys]['type'].lower() +' '+totalsentence[temp]
                    if strr in wordtag2.keys():
                        wordtag2[strr][0]+=1
                    else:
                        wordtag2[strr].append(1)
                        #print("---------------",wordtag2)
                    flag=1
                    break
            else:
                temp=i+k
                if temp>totalleng-1:
                    break
                if len(dictionaryMatch[new_dict[keys]['type']])==0:
                    break
                elif len(dictionaryMatch[totalsentence[temp]])==0:
                    k+=1
                    continue
                if (dictionaryMatch[new_dict[keys]['type']]) == (dictionaryMatch[totalsentence[temp]]):
                    #print('match')
                    strr=new_dict[keys]['type'].lower() +' '+totalsentence[temp]
                    if strr in wordtag2.keys():
                        wordtag2[strr][0]+=1
                    else:
                        wordtag2[strr].append(1)
                        #print("---------divya------",wordtag2)
                    flag=1
                    break
      
        if flag==0:
        
            #print(keys)
            pass
        else:
            #print(keys)
            i+=k+1
    return wordtag2

def regexchecker(str2):
    ok=re.match("\/[\S]*\/",str2)
    if ok is not None:
        return True
    else:
        return False
def checkforsecondhead(str3):
    ok=re.match("//[\S\s]*",str3)
    if ok is not None:
        return True
    else:
        return False

def checkForonlySymbols(str2):    
    ok=re.match("[\S\s]*[A-Za-z0-9][\S\s]*",str2)
    if ok is not None:
        return False
    else:
        return True
def checkForCommentStart(str3):
    ok=re.match("/\*[\S\s]*",str3)
    if ok is not None:
        return True
    else:
        return False
def checkForCommentend(str3):
    ok=re.match("[\S\s]*\*/$",str3)
    if ok is not None:
        return True
    else:
        return False
def checkforastring(str2):
    ok=re.match("\S*\'[\S]*\'\S*?",str2)
    ok2=re.match("\S*\"[\S]*\"\S*?",str2)
    if ok is not None or ok2 is not None:
        return True
    else:
        return False  
    

with open('nlp_project_all_tags', 'rb') as handle:
    all_tags = pickle.load(handle)
   
with open('nlp_pr_big.pickle', 'rb') as handle:
    bigram_dict = pickle.load(handle)

   
with open('nlp_project_transition', 'rb') as handle:
    transition= pickle.load(handle)


with open('nlp_project_emission', 'rb') as handle:
    emission= pickle.load(handle)
   
    
print('done')

unique_tags=[]
for key,value in all_tags.items():
    unique_tags.append(key)
   
    
    
def nxt_word_bi(word1):
    bi_ans=[] 
    #print("-------------------------------------bi----------------------------------------")
    for key,value in bigram_dict.items():
        if((key[0]==word1)):
            bi_ans.append(key[1])
    return bi_ans

def find_next_tag_given_word(word):
    if word not in emission:
        return 'anything'
    dictionary=emission[word]
    tag=max(dictionary.items(), key=operator.itemgetter(1))[0]
    next_tag=max(transition[tag].items(), key=operator.itemgetter(1))[0]
    return next_tag
   
def HMM(observ):
    result=[]
    for obser1 in observ:
        arr=np.zeros((len(unique_tags), len(obser1)))
        j=0
        if obser1[j] in emission:
            diction=emission[obser1[j]]
            for i in range(1,len(arr)):
                if unique_tags[i] in diction:
                    arr[i][j]=diction[unique_tags[i]]
           
            for j in range(1,len(obser1)):
                diction=emission[obser1[j]]
                for i in range(1,len(arr)):
                    if unique_tags[i] in transition:
                        dic_obs=transition[unique_tags[i]]
                        prob_s=0
                        for k in range(1,len(arr)):
                            if unique_tags[k] in dic_obs:
                                prob_s=prob_s+(arr[k][j-1]*dic_obs[unique_tags[k]])
                    if unique_tags[i] in diction:
                        arr[i][j]=diction[unique_tags[i]]*prob_s
               
               
            end=len(obser1)-1
            final=0
            for i in range(len(arr)):
                final=final+arr[i][end]
            result.append((final))
    return result



def sentencemaker(f):
    k=f.readlines()
    senetence=[]
    flag=0
    sent=''
    flag2=1
    stack=[]
    for line in k:
        line=line.strip()
        if checkForCommentStart(line)==True:
            flag2=0
            continue
        if checkForCommentend(line)==True:
            flag2=1
            continue
        if checkforsecondhead(line)==True:
            flag2=1
            continue
        templine=line.split()
        line=''
        line=line.strip()
        if len(templine)>0:
            if regexchecker(templine[0])==True:
                line+='<regexxxxvar>'
            else:
                line+=templine[0]
            for ookk in range(1,len(templine)):
                if regexchecker(templine[ookk])==True:
                    line+='<regexxxxvar>'
                else:
                    line+=' '+templine[ookk]
        if line=='{' or line=='}':
            continue
        uniigrm = list(wordpunct_tokenize(line))
        lengt=len(uniigrm)
        if flag2==1:
            if lengt>0:
                    if ';' in uniigrm[lengt-1]:
                       if flag==0:  
                            sent='<s>'
                       for i in uniigrm:
                            if '\'' in i or '\"' in i :
                                if len(stack)==0:
                                    pass
                                else:
                                    top=len(stack)-1
                                for q in i:
                                    if len(stack)==0:
                                        if q=='\'':
                                            sent+=' '+'<vars>' 
                                            stack.append('\'')
                                        elif q=='\"':
                                            sent+=' '+'<vars>' 
                                            stack.append('\"')
                                        
                                    else:
                                        top=len(stack)-1
                                        if top>=0 and (q=='\'' and stack[top]=='\'') or (q=='\"' and stack[top]=='\"'):
                                            stack.pop()
                                            top=top-1
                                        else:
                                            if q=='\'':
                                                stack.append('\'')
                                            elif q=='\"':
                                                stack.append('\"')
                            if len(stack)==0:
                                i=i.lower()
                                if i=='_':
                                    sent+=' '+i
                                elif i == 'in':
                                    temp=sent.split()
                                    if temp[len(temp)-3]=='for' or temp[len(temp)-2]=='if' :
                                        pass
                                    else:
                                         sent+=' '+i
                                elif checkForonlySymbols(i)==False:
                                   sent+=' '+i
                       sent+=' '+'</s>'
                       countofwords=sent.split()
                       if len(countofwords)>2:
                           senetence.append(sent)
                       sent=''
                       flag=0
                       stack=[]
                    else:
                        if flag==0:
                            flag=1
                            sent='<s>'
                        for i in uniigrm:
                            if '\'' in i or '\"' in i :
                                if len(stack)==0:
                                    sent+=' '+'<vars>' 
                                else:
                                    top=len(stack)-1
                                for q in i:
                                    if len(stack)==0:
                                        if q=='\'':
                                            stack.append('\'')
                                        elif q=='\"':
                                            stack.append('\"')
                                    else:
                                        top=len(stack)-1
                                        if top>=0 and (q=='\'' and stack[top]=='\'') or (q=='\"' and stack[top]=='\"'):
                                            stack.pop()
                                            top=top-1
                                        else:
                                            if q=='\'':
                                                stack.append('\'')
                                            elif q=='\"':
                                                stack.append('\"')
                            if len(stack)==0:
                                i=i.lower()
                                if i=='_':
                                    sent+=' '+i
                                elif i == 'in':
                                    temp=sent.split()
                                    if temp[len(temp)-3]=='for' or temp[len(temp)-2]=='if' :
                                        pass
                                    else:
                                         sent+=' '+i
                                elif checkForonlySymbols(i)==False:
                                   sent+=' '+i
    if len(sent)>0:
        sent+=' '+'</s>'
        senetence.append(sent)
    return senetence


training_filename=pd.read_csv('/home/nikhil/Documents/NLP project/js_dataset/programs_eval.txt', error_bad_lines=False,header=None)
dictionary=defaultdict(list)
select1000files=training_filename
path='/home/nikhil/Documents/NLP project/js_dataset/'
select1000files=np.array(select1000files)

q=0
p=0
def worddictionary(i,ts):
    k=0
    with open("/home/nikhil/Documents/NLP project/js_dataset/programs_eval.json", 'r',encoding='latin-1') as f:
       for line in f:
             if k==i:
                 data = json.loads(line)
                 new_dict = getJsonInfo(data)
                 wordtag=wordtdagconvertor(new_dict,ts)
             k+=1
             if k==(i+1):
                 break
    return wordtag
   
newA = sorted(bigram_dict.items(), key=operator.itemgetter(1), reverse=True)
sorted_dict = collections.OrderedDict(newA)

totalcountt=0
predictedcount=0        
for i in range(10000):
        totalsentence=[]
        tempfilename=str(select1000files[i][0]).strip()
        #file='/home/nikhil/Desktop/Untitled Document.js'
        file=path+tempfilename
        f = open(file , encoding='latin-1')
        senetence=sentencemaker(f)
        #print('part1',senetence)
        for w in senetence:
            w=w.split()
            for k in w:
                if k=='<s>' or k=='</s>':
                    pass
                else:
                    totalsentence.append(k)
        data = ""
        wordtag =worddictionary(i,totalsentence)
        tot=0
        for i in wordtag.keys():
            tot+=wordtag[i][0]
            #print(wordtag[i][0])
        wordlist=[]
        #print(wordtag)
        for lkk in wordtag.keys():
            kjj=lkk.split()
            if len(kjj)>1:
                 wordlist.append(kjj[1])
        #print(wordlist)
        newsent=''

        #print(senetence)
        for tk in senetence:
            for gk in tk.split():
                if gk=='<s>' or gk=='</s>':
                     if len(newsent)==0:
                            newsent+=gk
                     else:
                            newsent+=' '+gk
                else:
                    if len(newsent)==0:
                        if gk in wordlist:
                            newsent+=gk
                        else:
                            newsent+=' <unkk>'
                    else:
                         if gk in wordlist:
                             newsent+=' '+gk
                         else:
                            newsent+=' <unkk>'
        okkj=0
        newsent=newsent.split()
        while okkj < len(newsent):
            word1=newsent[okkj]
            #print(word1,okkj)
            if newsent[okkj]=='<s>' or newsent[okkj]=='</s>':
                okkj+=1
                continue
            else:
                 if newsent[okkj]=='<unkk>' or  newsent[okkj+1]=='<unkk>':
                     okkj+=1
                     continue
                 else:
                     nexttag=find_next_tag_given_word(newsent[okkj])
                     if nexttag=='variabledeclarator':
                         if newsent[okkj+1]=='<vars>':
                             predictedcount+=1
                             totalcountt+=1
                         else:
                             totalcountt+=1
                     elif nexttag=='literalnumber':
                         if newsent[okkj+1].isdigit()==True:
                             predictedcount+=1
                             totalcountt+=1
                         else:
                             totalcountt+=1
                     elif nexttag=='literalregexp':
                         if newsent[okkj+1]=='regexxxxvar':
                             predictedcount+=1
                             totalcountt+=1
                         else:
                             totalcountt+=1
                     elif nexttag in dictionarymatch2.keys() and newsent[okkj] in dictionarymatch2.keys():
                         if dictionarymatch2[nexttag]==dictionarymatch2[newsent[okkj]]:
                             predictedcount+=1
                             totalcountt+=1
                         else:
                             totalcountt+=1
                     else:
                        expec_words=nxt_word_bi(word1)
                        #print('edd',expec_words)
                        expected_words=[]
                        for word in expec_words:
                            if word in emission:
                                expected_words.append(word)  
                        expected_words=expected_words[:10]  
                        #print(expected_words)
                        observ=[]
                        for eps in expected_words:
                            if eps not in exclude:
                                obs=[]
                                obs.append(word1)
                                obs.append(eps)
                                observ.append(obs)
                        result=HMM(observ)
                        #print('result',observ)
                        inde=sorted(range(len(result)), key=lambda poo: result[poo], reverse=True)[:14]
                        #print('dkjjd',inde)
                        #inde=result.index(max(result))
                        lis=[]
                        for qe in inde:
                            lis.append(observ[qe][1])
                        taggg=0
                        for qp in lis:
                            if qp == newsent[okkj+1]:
                                taggg=1
                                break
                        if taggg==1:
                            predictedcount+=1
                            totalcountt+=1
                        else:
                             totalcountt+=1
            okkj+=1
           
            #print(okkj,totalcountt)
        p=p+1
        print('After File Number',p,'  ',round(predictedcount/totalcountt*100,2))
        #print(totalsentence)
    


print(round(predictedcount/totalcountt*100,2))



    
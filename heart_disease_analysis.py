#!/usr/bin/env python
# coding: utf-8

# # Imports
# 
# Zona de importação das bibliotecas utilizadas neste trabalho.

# In[1]:


import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import seaborn as sns
sns.set(color_codes=True)
from matplotlib import rcParams
from matplotlib import pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
import scipy as sp
import scipy.stats as stats
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


# # Importação do ficheiro
# 
# É utilizado a biblioteca "Pandas" por se proceder à importação do ficheiro "cardio.csv".
# Neste ficheiro estão identificados indivídiuos, com determinados dados, que indicam se tem alguma doença cardiovascular ou não.
# É feita uma primeira visualização de uma pequena amostra das primeiras linhas do dataframe, para ter uma ideia do conteúdo dos dados.

# In[2]:


heart_disease = pd.read_csv('cardio_train.csv')
heart_disease.head()


# # Limpeza dos dados
# 
# Nesta fase é realizada o tratamento e a limpeza dos dados.

# Como os dados da coluna "age" estão em dias, irá ser convertido para anos, para ser mais fácil de entender a idade de cada indivíduo. Ao executar esta converção, é necessário também converter o tipo de dados da coluna para o tipo inteiro (numpy int64). 
# Remover-se-á a coluna "id" porque o index da biblioteca Pandas já tem a mesma finalidade que este campo, que é identificar univocamente cada linha.

# In[3]:


heart_disease['age'] = round(heart_disease['age'] * 0.00273790926)
heart_disease = heart_disease.astype({'age': np.int64})

del heart_disease['id']
heart_disease.head()


# In[4]:


heart_disease.info()


# Podemos evidenciar que o dataframe tem 70000 dados, tem 12 colunas, todos os dados são do tipo numpy int64, à excepção dos dados da coluna "weight", que é do tipo numpy float64.

# In[5]:


heart_disease.isnull().any()


# O dataframe não tem quaisqueres missing values, i.e., None ou NaN ou Null.
# Por esse motivo não é necessário proceder ao tratamento de missing values.

# In[6]:


heart_disease.describe()


# Com a informação acima, podemos ter uma noção geral do dataframe.

# In[7]:


heart_disease['gender'].value_counts()


# Existem mais mulheres ("gender" = 1) do que homens ("gender" = 2).

# In[8]:


sns.boxplot(heart_disease['age'])


# Verifica-se que a média de idades é aproximadamente 54 anos, o mínimo é 30 anos (apenas com 1 indivíduo) e o máximo 65 anos, logo podemos afirmar que temos uma amostra de indivíduos entre os 30 e os 65 anos.

# Como apenas temos 1 indivíduo com 30 anos, iremos removê-lo.

# In[9]:


heart_disease.drop(heart_disease[(heart_disease['age'] == 30)].index,inplace=True)


# In[10]:


sns.boxplot(heart_disease['height'])


# A altura varia entre 55 centimetros e 2 metros e 50 centimetros, sendo que a média é de 1 metro e 64 centimetros.

# In[11]:


sns.boxplot(heart_disease['weight'])


# O peso varia entre 10 kg e os 200 kg, em que a média são 74 kg.

# Como se pôde constactar existem outliers, mais propriamente em relação à altura de 55 centimetros e 2 metros e 50 centimetros e o peso de 10 kg.
# Por esse facto ir-se-á proceder à remoção dos mesmos.

# In[12]:


heart_disease.drop(heart_disease[(heart_disease['height'] > heart_disease['height'].quantile(0.99999)) | (heart_disease['height'] < heart_disease['height'].quantile(0.0004))].index,inplace=True)
heart_disease.drop(heart_disease[(heart_disease['weight'] < heart_disease['weight'].quantile(0.001))].index,inplace=True)


# In[13]:


sns.boxplot(heart_disease['height'])


# In[14]:


sns.boxplot(heart_disease['weight'])


# Existem também outliers em relação à pressão arterial, valores em que a pressão diastólica ("ap_lo") é maior que a sistólica ("ap_hi"), valores negativos em ambos, valores demasiado elevados e demasiado baixos, e por isso vamos removê-los.

# In[15]:


outliers = (heart_disease['ap_lo'] > heart_disease['ap_hi']).sum()

print("Existem {} casos em que pressão diastólica é maior que a sistólica".format(str(outliers)))


# In[16]:


outliers = (heart_disease['ap_lo'] < 0).sum()
print("Existe {} caso em que pressão diastólica é negativa".format(str(outliers)))


# In[17]:


outliers = (heart_disease['ap_hi'] < 0).sum()
print("Existem {} casos em que pressão diastólica é negativa".format(str(outliers)))


# In[18]:


heart_disease.drop(heart_disease[(heart_disease['ap_hi'] > heart_disease['ap_hi'].quantile(0.975)) | (heart_disease['ap_hi'] < heart_disease['ap_hi'].quantile(0.025))].index,inplace=True)
heart_disease.drop(heart_disease[(heart_disease['ap_lo'] > heart_disease['ap_lo'].quantile(0.975)) | (heart_disease['ap_lo'] < heart_disease['ap_lo'].quantile(0.025))].index,inplace=True)


# In[19]:


blood_pressure = heart_disease.loc[:,['ap_lo','ap_hi']]
sns.boxplot(x = 'variable',y = 'value',data = blood_pressure.melt())


# Vamos agora averiguar se existem duplicados.

# In[20]:


print("Existem {} duplicados".format(heart_disease.duplicated().sum()))


# Vamos visualizar alguns desses duplicados.

# In[21]:


duplicated = heart_disease[heart_disease.duplicated(keep=False)]
duplicated = duplicated.sort_values(by=['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio'], ascending= True)
duplicated.head(heart_disease.duplicated().sum())


# Como os duplicados não ajudam para a nossa análise são removidos.

# In[22]:


heart_disease.drop_duplicates(inplace=True)


# Após o tratamento e limpeza de dados ainda nos fica a faltar variáveis importantes neste tipo de caso de estudo, que são o Indice de Massa Corporal e a Pressão do Pulso.

# In[23]:


heart_disease['bmi'] = heart_disease['weight'] / ((heart_disease['height'] * 0.01) ** 2)
heart_disease['pulse_pressure'] = heart_disease['ap_hi'] - heart_disease['ap_lo']

heart_disease.head()


# # PairPlot
# 
# Com o pairplot vai ser possivel explorar correlações entre dados multidimensionais.

# In[24]:


sns.pairplot(heart_disease, hue='cardio', aspect=0.5)


# # HeatMap
# 
# O heatmap dá-nos a visualização de quais as variáveis com maior correlação.
# Neste caso queremos saber quais as variáveis com maior correlação com a variável "cardio".

# In[25]:


f, ax = plt.subplots(figsize = (15,15))
mask = np.zeros_like(heart_disease.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(heart_disease.corr(), mask=mask, annot=True, fmt=".3f", linewidths=0.5, ax=ax, cmap='RdYlGn')


# Com isto podemos afirmar que pressão arterial (ap_hi e ap_lo), pressão do pulso têm uma maior correlação com cardio, i.e. têm maior relação com doenças cardiovasvulares. Existe uma grande correlação entre as pressões arteriais (ap_hi e ap_lo). Existe correlação entre as beber e fumar e entre colesterol e açúcar no sangue. Existe uma grande correlação entre peso e indice de massa corporal. Existe uma grande correlação entre a pressão sistólica ("ap_hi") e a pressão do pulso.

# # GroupBy: Split, Apply, Combine

# Iremos daqui em diante analisar os dados mais ao detalhe.
# Com isto, iremos começar por agrupar "age" com "gender".

# In[26]:


age_gender_bmi = heart_disease.groupby(['age', 'gender']).agg({'bmi': np.mean})


# In[27]:


age_gender_bmi = age_gender_bmi[age_gender_bmi.bmi > 25]
age_gender_bmi


# In[28]:


age_gender_bmi.pivot_table('bmi', index='age', columns='gender', aggfunc='mean')


# In[29]:


heart_disease.pivot_table('bmi', index='age', columns='gender', aggfunc='mean').plot()
plt.ylabel('Mean bmi per age')


# Aplicou-se a média de "bmi" por "age" e "gender".
# Em todas as idades e géneros têm uma média de Indice de Massa Corporal acima do normal, trata-se por normal Indice de Massa Corporal < 25.
# As mulheres têm uma média de Indice de Massa Corporal superior à dos homens.
# As mulheres com mais idade têm maior média de Indice de Massa Corpora.

# In[30]:


age_gender_blood_pressure = heart_disease.groupby(['age', 'gender']).agg({'ap_hi': np.mean, 'ap_lo': np.mean})


# In[31]:


age_gender_blood_pressure = age_gender_blood_pressure[age_gender_blood_pressure.ap_hi > 120]
age_gender_blood_pressure = age_gender_blood_pressure[age_gender_blood_pressure.ap_lo > 80]
age_gender_blood_pressure


# In[32]:


heart_disease.pivot_table('pulse_pressure', index='age', columns='gender', aggfunc='mean')


# In[33]:


heart_disease.pivot_table('pulse_pressure', index='age', columns='gender', aggfunc='mean').plot()
plt.ylabel('Mean pulse pressure per age')


# Aplicou-se a média de "ap_hi" e "ap_lo" por "age" e "gender".
# Com esta pequena análise verifica-se que antes dos 41 anos não foram encontrados indivíduos com média de pressão arterial acima do normal, compreende-se acima do normal como pressão sistólica > 120 e pressão diastólica > 80.
# Dos 41 aos 46 anos só os homens têm média de pressão arterial acima do normal.
# A partir dos 47 anos tanto homens como mulheres têm média de pressão arterial acima do normal.
# Indivíduos com mais idade têm mais média de pressão arterial.
# Os homens no geral têm média de pressão arterial superior ao das mulheres.

# In[34]:


rcParams['figure.figsize'] = 14, 7
sns.countplot(x='age', hue='cardio', data = heart_disease, palette="Set1")


# É visível que com o decorrer da idade, existe maior probabilidade de ter uma doença cardiovascular.
# A partir dos 55 anos existem mais indivíduos com doenças cardiovasculares do que sem doenças cardiovasculares.

# In[35]:


heart_disease_categories = pd.melt(heart_disease, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active'])
sns.catplot(x="variable", hue="value", col="cardio", data=heart_disease_categories, kind="count")


# Com esta informação ficamos com a ideia que pessoas com doenças cardiovasculares apresentam maiores níveis de colesterol fora do normal ("Cholesterol" = 2 ou "Cholesterol" = 3), maiores níveis de açucar no sangue ("gluc" = 2 ou "gluc" = 3) e fazem menos actividade física. Em relação aos níveis de consumo de tabaco e alcóol, não parece existir diferença.

# In[36]:


heart_disease_categories = pd.melt(heart_disease, id_vars=['gender'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio'])
sns.catplot(x="variable", hue="value", col="gender", data=heart_disease_categories, kind="count")


# Apesar de existirem mais mulheres do que homens, é visível que os homens consomem mais tabaco e alcóol que as mulheres. Em relação às doenças cardiovasculares, nas mulheres existem mais com doenças cardiovasculares e nos homens os valores são muito similares, apesar dos níveis serem praticamente quase iguais, dentro de cada género. Podemos visualizar abaixo todos os factos que mencionei.

# In[37]:


heart_disease.groupby('gender')['smoke'].sum()


# In[38]:


heart_disease.groupby('gender')['alco'].sum()


# In[39]:


# womens
womens = (heart_disease['gender'] == 1).sum()
cholesterol = ((heart_disease['gender'] == 1) & (heart_disease['cholesterol'] != 1)).sum()
gluc = ((heart_disease['gender'] == 1) & (heart_disease['gluc'] != 1)).sum()
active = ((heart_disease['gender'] == 1) & (heart_disease['active'] == 1)).sum()

percentage_cholesterol = round(((cholesterol * 100)/womens),1)
percentage_gluc = round(((gluc * 100)/womens),1)
percentage_active = round(((active * 100)/womens),1)

print("Percentagem de mulheres com colesterol fora do normal: " + str(percentage_cholesterol) + "%")
print("Percentagem de mulheres com níveis de açúcar no sangue fora do normal: " + str(percentage_gluc) + "%")
print("Percentagem de mulheres que praticam actividades físicas: " + str(percentage_active) + "%")


# In[40]:


# mans
mans = (heart_disease['gender'] == 2).sum()
cholesterol = ((heart_disease['gender'] == 2) & (heart_disease['cholesterol'] != 1)).sum()
gluc = ((heart_disease['gender'] == 2) & (heart_disease['gluc'] != 1)).sum()
active = ((heart_disease['gender'] == 2) & (heart_disease['active'] == 1)).sum()

percentage_cholesterol = round(((cholesterol * 100)/mans),1)
percentage_gluc = round(((gluc * 100)/mans),1)
percentage_active = round(((active * 100)/mans),1)

print("Percentagem de homens com colesterol fora do normal: " + str(percentage_cholesterol) + "%")
print("Percentagem de homens com níveis de açúcar no sangue fora do normal: " + str(percentage_gluc) + "%")
print("Percentagem de homens que praticam actividades físicas: " + str(percentage_active) + "%")


# Existem mais mulheres do que homens com colesterol fora do normal e níveis de açúcar no sangue fora do normal, apesar dos valores estarem próximos.

# In[41]:


catplot = sns.catplot(x="gender", y="bmi", hue="alco", col="cardio", data=heart_disease ,kind="box", height=10, aspect=.6)
axes = catplot.axes
axes[0,0].set_ylim(0,80)


# Mulheres que bebem têm mais riscos de ter doenças cardiovasculares do que os homens.

# Vamos ver quais as médias das variáveis em relação ao "cardio".

# In[42]:


heart_disease.groupby('cardio')[['age', 'gender', 'height', 'weight', 'bmi', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']].mean()


# Em média quem tem doenças cardiovasculares apresenta mais idade, mais peso, mais Indice de Massa Corporal, mais pressão arterial, mais colesterol, mais açúcar no sangue e faz menos actividade física.

# Como já visto, as variáveis com maior impacto são "ap_hi", "ap_lo" e "pulse_pressure".
# Com a ajuda do pairplot e do heatmap iremos descobrir quais as restantes variáveis que têm correlação com estas.
# - "ap_hi" correlaciona-se com "pulse_pressure", "bmi", "ap_lo" e "weight";
# - "ap_lo" correlaciona-se com "ap_hi", "bmi" e "weight";
# - "pulse_pressure" correlaciona-se com "ap_hi".

# # Regressão Linear
# 
# É realizada regressão linear entre as variáveis descritas.
# É calculado o coeficiente de determinação, correlação de pearson e spearman entre as mesmas. 

# Função que recebe 2 variáveis, e a partir dos valores das mesmas faz a regressão linear e calcula o coeficiente de determinação, correlação pearson e spearman.

# In[43]:


def linear_regression(field_1, field_2):
    var_1 = heart_disease[field_1]
    var_2 = heart_disease[field_2]
    
    slope, intercept, r_value, p_value, std_err = sp.stats.linregress(var_1, var_2)
    heart_disease.plot(x=field_1, y=field_2, kind='scatter')
    plt.plot(var_1,var_1*slope+intercept,'r')
    plt.show()
    print ("r-squared : {}".format(r_value**2))
    print ("Pearson correlation : {}".format(pearsonr(var_1, var_2)[0]))
    print ("Spearman correlation : {}".format(spearmanr(var_1, var_2)[0]))


# In[44]:


#ap_hi
linear_regression('ap_hi', 'pulse_pressure')
linear_regression('ap_hi', 'bmi')
linear_regression('ap_hi', 'ap_lo')
linear_regression('ap_hi', 'weight')


# In[45]:


#ap_lo
linear_regression('ap_lo', 'ap_hi')
linear_regression('ap_lo', 'bmi')
linear_regression('ap_lo', 'weight')


# In[46]:


#pulse_pressure
linear_regression('pulse_pressure', 'ap_hi')


# Com esta a informação, permite-nos tirar algumas conclusãos, tais como:

# O valor do campo "ap_hi" aumenta significativamente com o aumento dos campos "pulse_pressure" e "ap_lo" e aumenta ligeiramente com o aumento dos campos "bmi" e "weight".

# O valor do campo "ap_lo" aumenta significativamente com o aumento do campo "ap_hi" e aumenta ligeiramente com o aumento dos campos "bmi" e "weight".

# O valor do campo "pulse_pressure" aumenta significativamente com o aumento do campo "ap_hi".

# # Teste Kolmogorov-Smirnov, CDF e ECDF

# In[47]:


def distribution(field):
    var = heart_disease[field]
    sns.kdeplot(var, shade=True)
    return


# In[48]:


cdfs = [
    "norm",            #Normal (Gaussian)
    "alpha",           #Alpha
    "anglit",          #Anglit
    "arcsine",         #Arcsine
    "beta",            #Beta
    "betaprime",       #Beta Prime
    "bradford",        #Bradford
    "burr",            #Burr
    "cauchy",          #Cauchy
    "chi",             #Chi
    "chi2",            #Chi-squared
    "cosine",          #Cosine
    "dgamma",          #Double Gamma
    "dweibull",        #Double Weibull
    "erlang",          #Erlang
    "expon",           #Exponential
    "exponweib",       #Exponentiated Weibull
    "exponpow",        #Exponential Power
    "fatiguelife",     #Fatigue Life (Birnbaum-Sanders)
    "foldcauchy",      #Folded Cauchy
    "f",               #F (Snecdor F)
    "fisk",            #Fisk
    "foldnorm",        #Folded Normal
    "frechet_r",       #Frechet Right Sided, Extreme Value Type II
    "frechet_l",       #Frechet Left Sided, Weibull_max
    "gamma",           #Gamma
    "gausshyper",      #Gauss Hypergeometric
    "genexpon",        #Generalized Exponential
    "genextreme",      #Generalized Extreme Value
    "gengamma",        #Generalized gamma
    "genlogistic",     #Generalized Logistic
    "genpareto",       #Generalized Pareto
    "genhalflogistic", #Generalized Half Logistic
    "gilbrat",         #Gilbrat
    "gompertz",        #Gompertz (Truncated Gumbel)
    "gumbel_l",        #Left Sided Gumbel, etc.
    "gumbel_r",        #Right Sided Gumbel
    "halfcauchy",      #Half Cauchy
    "halflogistic",    #Half Logistic
    "halfnorm",        #Half Normal
    "hypsecant",       #Hyperbolic Secant
    "invgamma",        #Inverse Gamma
    "invgauss",         #Inverse Normal
    "invweibull",      #Inverse Weibull
    "johnsonsb",       #Johnson SB
    "johnsonsu",       #Johnson SU
    "laplace",         #Laplace
    "logistic",        #Logistic
    "loggamma",        #Log-Gamma
    "loglaplace",      #Log-Laplace (Log Double Exponential)
    "lognorm",         #Log-Normal
    "lomax",           #Lomax (Pareto of the second kind)
    "maxwell",         #Maxwell
    "mielke",          #Mielke's Beta-Kappa
    "nakagami",        #Nakagami
    "ncx2",            #Non-central chi-squared
    "nct",             #Non-central Student's T
    "pareto",          #Pareto
    "powerlaw",        #Power-function
    "powerlognorm",    #Power log normal
    "powernorm",       #Power normal
    "rdist",           #R distribution
    "reciprocal",      #Reciprocal
    "rayleigh",        #Rayleigh
    "rice",            #Rice
    "recipinvgauss",   #Reciprocal Inverse Gaussian
    "semicircular",    #Semicircular
    "t",               #Student's T
    "triang",          #Triangular
    "truncexpon",      #Truncated Exponential
    "truncnorm",       #Truncated Normal
    "tukeylambda",     #Tukey-Lambda
    "uniform",         #Uniform
    "vonmises",        #Von-Mises (Circular)
    "wald",            #Wald
    "weibull_min",     #Minimum Weibull (see Frechet)
    "weibull_max",     #Maximum Weibull (see Frechet)
    "ksone",           #Kolmogorov-Smirnov one-sided (no stats)
    "kstwobign"]       #Kolmogorov-Smirnov two-sided test for Large N


# In[49]:


def sk_test(field):
    var = heart_disease[field]
    sorted_field = np.sort(var)
    p_value = 0.05
    print(str(field.upper()) + "\n")
    for cdf in cdfs:
        parameters = eval("sp.stats."+cdf+".fit(sorted_field)")
        D, p = sp.stats.kstest(sorted_field, cdf, args=parameters)
        print ("Distribution: " + str(cdf.ljust(16)) + ("p: "+str(p)).ljust(25)+"D: "+str(D))
        if p < p_value:
            print("É rejeitada a hipótese nula\n")
        else:
            print("Não é rejeitada a hipótese nula\n")


# In[50]:


def sk_plot_norm(field):
    var = heart_disease[field]
    length = len(var)
    mu = sp.mean(var)
    plt.figure(figsize=(12, 7))
    plt.plot(np.sort(var), np.linspace(0, 1, length), linewidth=3.0)
    plt.plot(np.sort(stats.norm.rvs(loc=mu, scale=5, size=length)), np.linspace(0, 1, length), linewidth=3.0, color="r")
    plt.legend('top right')
    plt.legend(['CDF', 'ECDF'])
    plt.title(field)
    plt.show()


# In[51]:


def cdf(field):
    var = heart_disease[field]
    var.hist(cumulative = True)
    sorted_field = np.sort(var)
    plt.step(sorted_field, np.arange(sorted_field.size), linewidth=5.0, color="r")
    plt.show()


# In[52]:


def sk_test_2samp(field):
    var = heart_disease[field]
    length = len(var)
    half_length = int((length/2)-0.5)
    first_half = var.loc[half_length:]
    second_half = var.loc[:half_length]
    p_value = 0.05
    sk_2samp = stats.ks_2samp(first_half, second_half)
    print (str(field.upper()) + "\n\n" + str(sk_2samp))
    if sk_2samp[1] < p_value:
        print("É rejeitada a hipótese nula")
    else:
        print("Não é rejeitada a hipótese nula")
        
    return first_half, second_half


# In[53]:


def ks_plot_comp_cdf(field, first_half, second_half):
    mu_1 = sp.mean(first_half)
    mu_2 = sp.mean(second_half)
    plt.figure(figsize=(12, 7))
    
    if mu_1 > mu_2:
        diff_mu = mu_1 - mu_2
        plt.plot(np.sort(first_half), np.linspace(0, 1, len(first_half), endpoint=False), linewidth=3.0)
        plt.plot(np.sort(second_half + diff_mu), np.linspace(0, 1, len(second_half), endpoint=False), linewidth=3.0, color="r")
    if mu_1 < mu_2:
        diff_mu = mu_2 - mu_1
        plt.plot(np.sort(first_half + diff_mu), np.linspace(0, 1, len(first_half), endpoint=False), linewidth=3.0)
        plt.plot(np.sort(second_half), np.linspace(0, 1, len(second_half), endpoint=False), linewidth=3.0, color="r")
        
    plt.legend('top right')
    plt.legend(['First half', 'Second half'])
    plt.title('Comparing ' + str(field) + ' first half and fecond half CDFs')
    plt.xticks([])
    plt.show()


# In[54]:


def ks_plot_comp_ecdf(field, first_half, second_half):
    mu_1 = sp.mean(first_half)
    mu_2 = sp.mean(second_half)
    plt.figure(figsize=(12, 7))
    
    if mu_1 > mu_2:
        diff_mu = mu_1 - mu_2
        plt.plot(np.sort(stats.norm.rvs(loc=first_half, scale=5, size=len(first_half))), np.linspace(0, 1, len(first_half), endpoint=False), linewidth=3.0, color="g")
        plt.plot(np.sort(stats.norm.rvs(loc=sp.mean(second_half + diff_mu), scale=5, size=len(second_half))), np.linspace(0, 1, len(second_half), endpoint=False), linewidth=3.0, color="orange")
    if mu_1 < mu_2:
        diff_mu = mu_2 - mu_1
        plt.plot(np.sort(stats.norm.rvs(loc=sp.mean(first_half + diff_mu), scale=5, size=len(first_half))), np.linspace(0, 1, len(first_half), endpoint=False), linewidth=3.0, color="g")
        plt.plot(np.sort(stats.norm.rvs(loc=second_half, scale=5, size=len(second_half))), np.linspace(0, 1, len(second_half), endpoint=False), linewidth=3.0, color="orange")
        
    plt.legend('top right')
    plt.legend(['First half', 'Second half'])
    plt.title('Comparing ' + str(field) + ' first half and fecond half ECDFs')
    plt.xticks([])
    plt.show()


# In[55]:


distribution('ap_hi')


# In[56]:


sk_test('ap_hi')


# In[57]:


sk_plot_norm('ap_hi')


# In[58]:


cdf('ap_hi')


# In[59]:


first_half, second_half = sk_test_2samp('ap_hi')


# In[60]:


ks_plot_comp_cdf('ap_hi', first_half, second_half)


# In[61]:


ks_plot_comp_ecdf('ap_hi', first_half, second_half)


# In[62]:


distribution('ap_lo')


# In[63]:


sk_test('ap_lo')


# In[64]:


sk_plot_norm('ap_lo')


# In[65]:


cdf('ap_lo')


# In[66]:


first_half, second_half = sk_test_2samp('ap_lo')


# In[67]:


ks_plot_comp_cdf('ap_lo', first_half, second_half)


# In[68]:


ks_plot_comp_ecdf('ap_lo', first_half, second_half)


# In[69]:


distribution('pulse_pressure')


# In[70]:


sk_test('pulse_pressure')


# In[71]:


sk_plot_norm('pulse_pressure')


# In[72]:


cdf('pulse_pressure')


# In[73]:


first_half, second_half = sk_test_2samp('pulse_pressure')


# In[74]:


ks_plot_comp_cdf('pulse_pressure', first_half, second_half)


# In[75]:


ks_plot_comp_ecdf('pulse_pressure', first_half, second_half)


# Dada esta informação concluí-se que:

# Visto que o valor p assume sempre o valor 0, i.e., valores tão baixos que arrendodados dão 0, é então rejeitada a hipotése nula. 
# As distribuições de dados não seguem qualquer distribuição mencionada.

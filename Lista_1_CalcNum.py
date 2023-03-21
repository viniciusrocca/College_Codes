# Grupo 8

## Thales Henrique Nogueira, RA: 11201811829
## Vinícius Zilio Rocca, RA: 11201810013

from scipy.optimize import curve_fit
import numpy as np
from tabulate import tabulate
import time

#Funções utilizadas ao longo das tarefas

#Esta função calcula uma integral utilizando a regra dos trapézios
def trapezoidal_rule(a,b,m,func):
    h = (b-a)/m
    t_m = 0.5*(func(a)+func(b))
    for i in range(1,m):
        x_i = a + i*h
        t_m = t_m + func(x_i)
        
    return t_m*h

#Esta função calcula uma integral utilizando a regra dos trapézios. Porém esta está voltada para funcionar dentro de um looping da tarefa 3
def trapezoidal_rule_Tarefa_3(a,b,m,k,func):
    h = (b-a)/m
    t_m = 0.5*(func(a,k)+func(b,k))
    for i in range(1,m):
        x_i = a + i*h
        t_m = t_m + func(x_i,k)
        
    return t_m*h

#Esta função calcula o valor de g(x) no ponto x
def g(x):
    return 2520*(5*np.power(x,3) - 10*np.power(x,2) + 6*x - 1)

#Esta função calcula o valor da primeira derivada de g(x) no ponto x
def d_g(x):
    return 2520*(15*np.power(x,2) - 20*x + 6)

#Esta função calcula o valor da segunda derivada de g(x) no ponto x
def d2_g(x):
    return 25200*(3*x-2)

#Esta função calcula o valor da função f(x) = 105 x^2(1-x)^4 no ponto x
def f1(x):
    return 105*np.power(x,2)*np.power(1-x,4)

#Esta função calcula a segunda derivada de f(x) no ponto x
def d2_f1(x):
    return 210*(15*np.power(x,2) -10*x + 1)*np.power((1-x),2)

#Esta função calcula o valor de F(\tau)  = int_0^\tau f(x)dx - 0.5 no ponto \tau
def fao(tau):
    return (trapezoidal_rule(0,tau,10000,f1) - 0.5)

#Esta função calcula o valor da função f(x) = x^{2k-1}
def f_Tarefa_3(x,k):
    return np.power(x,(2*k)-1)

#Esta função calcula o valor da integral de x^{2k-1} com limites de integração de a até b
def int_f_Tarefa_3(k,a,b):
    return (np.power(b,2*k) - np.power(a,2*k))/(2*k)

#Esta função calcula o valor da terceira derivada de f(x) = x^{2k-1}
def d3f_Tarefa_3(x,k):
    return (2*k-1)*(2*k-2)*(2*k-3)*np.power(x,2*k-4)

#Esta função calcula o número de Bernoulli B4
def b4(x_2,a,b,k,func):
    s = 5
    h = (b-a)/(s*np.power(2,k-1))
    return (x_2/np.power(h,4)) * (24/(d3f_Tarefa_3(b,k) - d3f_Tarefa_3(a,k)))

#Esta função utiliza o método de Newton para encontrar a raíz da função func com precisão \epsilon. Além disto ela também retorna as aproximações geradasnas iterações
def newton_method(a,b,alpha_0,n_max,epsilon,func,d_func):
    alpha = [alpha_0]
    n = 0
    while (func(alpha[n] + epsilon)*func(alpha[n] - epsilon) >= 0 and n <= n_max):
        alpha.append(alpha[n] - func(alpha[n])/d_func(alpha[n]))
        n = n + 1
    return alpha[n],alpha[1:]

#Esta função utiliza o método da bisseção para encontrar a raíz da função func com precisão \epsilon. Além disto ela também retorna as aproximações geradasnas iterações
def bisection_method(a,b,n_max,epsilon,func):
    inf = a
    sup = b
    alpha = [(a+b)/2]
    n=0
    while (func(alpha[n] + epsilon)*func(alpha[n] - epsilon) >= 0 and n <= n_max):
        if(func(inf)*func(alpha[n]) == 0):
            if(func(inf) == 0):
                return inf, alpha[1:]
            else:
                return alpha[n], alpha[1:]
            
        elif(func(inf)*func(alpha[n])< 0):
            sup = alpha[n]
            
        else:
            inf = alpha[n]
            
        alpha.append((sup+inf)/2) 
        n = n + 1
    return alpha[n], alpha[1:]

#Esta função imprime uma tabela contendo as aproximações geradas por um dos métodos acima
def approx_printer(alpha, fmt = "plain", formato = ".17f"):
    index = np.array(["i"],dtype = object)
    
    for j in range(1,len(alpha)+1):
        index = np.append(index,j)
        
    
    alpha = np.array(np.append(["alpha_i"],alpha), dtype = object)
        
    table = [index,alpha]
    print(tabulate([alpha], headers = index, tablefmt = fmt,numalign="center",floatfmt = formato))
    return

#Esta função imprime uma tabela em um arquivo .txt contendo as aproximações geradas por um dos métodos acima
def approx_printer_txt(alpha, f, fmt = "plain"):
    index = np.array(["i"],dtype = object)
    
    for j in range(1,len(alpha)+1):
        index = np.append(index,j)
        
    
    alpha = np.array(np.append(["alpha_i"],alpha), dtype = object)
        
    table = [index,alpha]
    print(tabulate([alpha], headers = index, tablefmt = fmt,numalign="center",floatfmt = ".17f"), file = f)
    return

#Esta função utiliza o método de Gauss para resolver um determinado sistema. Ela retorna a soluçao do sistema
def eliminacaoGauss(A, y, m, print_a = False, print_y = False):
  # 1ª parte (escalonamento na forma triangular superior)
    for j in range(0,m):
        k=j+1
        if (A[j][j] == 0):
            while (k!=m-1):
                if (A[k][j] != 0):
                    #Troca X
                    aux = A[k]
                    A[k] = A[j]
                    A[j] = aux
                    #Troca Y
                    aux = y[j]
                    y[j] = y[k]
                    y[k] = aux
                    print("Troca:",A) 
                    break
                elif(k==m-1):
                    print("Erro: A matriz A é singular")
                    return
                else:
                    k+=1
        
        for i in range(j+1,m):
            mu = -A[i][j]/A[j][j]
            y[i] = y[i]+mu*y[j]
            for l in range(j,m):
                A[i][l] = A[i][l]+mu*A[j][l]
    if print_a:
        print(A)
    if print_y:    
        print(y)
        
# Enconrando o valor das variáveis
    x= np.zeros(m)
    
    
    x[m-1] = y[m-1]/A[m-1][m-1]
    for i in range(m-2,-1, -1):
        x[i] = y[i]
        for k in range(i+1,m):
            x[i] = x[i]-A[i][k]*x[k]
        x[i] = x[i]/A[i][i]
    return x

def main():
    
    #####################Tarefa 1, item 1#############################################################
    
    #Valores de x
    x_values = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    text = ['x','g(x)',"g'(x)", "g''(x)"]

    #Tabela de dados
    table_data = np.array([ x_values, g(x_values),d_g(x_values),d2_g(x_values)] , dtype = object)
    table_data = np.insert(table_data,[0],[['x'],['g(x)'],["g'(x)"], ["g''(x)"]], axis = 1)
    
    #Imprimindo uma tabela com os valores de g(x), g'(x) e g''(x)
    print("Tarefa 1, Item 1")
    print("A tabela contendo todas as casas decimais é:")
    print(tabulate(table_data[1:],headers = table_data[:1][0],tablefmt='grid',numalign="right", floatfmt = ".17f"))
    print()
    print('A Tabela em sua versão de visualizacao facilitada:')
    print(tabulate(table_data,tablefmt='grid',numalign="right", floatfmt = ".1f"))
    print()
    
    #Escrevendo a tabela em um arquivo .txt
    f = open("lista_1_output.txt", "w")
    print('Tarefa 1, item 1', file = f)
    print(tabulate(table_data[1:],headers = table_data[:1][0],tablefmt='grid',numalign="right", floatfmt = ".17f"), file = f)
    print(file = f)
    f.close()
    
    
    #####################Tarefa 1, item 2 #############################################################
    # Contas da verificação I1 = [0.2, 0.3] e α0 = 0.3:
    print('Tarefa 1, item 2')
    print('Verificacao para I1 e alpha_0=0.3')
    print('g(0.3)*g(0.2) =', g(0.3)*g(0.2))
    print('alpha_1 =', 0.3-g(0.3)/d_g(0.3))
    print()
    
    # Contas da verificação I2 = [0.7, 0.8] e α0 = 0.7:
    print('Verificacao para I2 e alpha_0=0.7')
    print('g(0.7)*g(0.8) =', g(0.7)*g(0.8))
    print('alpha_1 =', 0.7-g(0.7)/d_g(0.7))
    print()

    #Escrevendo esses resultados em um arquivo .txt
    f = open("lista_1_output.txt", "a")
    print('Tarefa 1, item 2', file = f)
    print('Verificacao para I1 e alpha_0=0.3', file = f)
    print('g(0.3)*g(0.2) =', g(0.3)*g(0.2), file = f)
    print('alpha_1 =', 0.3-g(0.3)/d_g(0.3), file = f)
    print( file = f)
    print('Verificacao para I2 e alpha_0=0.7', file = f)
    print('g(0.7)*g(0.8) =', g(0.7)*g(0.8), file = f)
    print('alpha_1 =', 0.7-g(0.7)/d_g(0.7), file = f)
    print(file = f)
    f.close()
    
    
    #####################Tarefa 1, item 3#############################################################
    #Calculando x*_1:
    #Dados:
    a = 0.2
    b = 0.3
    alpha_0 = 0.3
    n_max = 1000
    epsilon = 1e-10
    
    #Utilizando o método de Newton:
    x_1 = newton_method(a,b,alpha_0,n_max,epsilon,g,d_g)
    
    
    #Calculando x*_2:
    #Dados
    a = 0.7
    b = 0.8
    alpha_0 = 0.7
    
    #Utilizando o método de Newton:
    x_2 = newton_method(a,b,alpha_0,n_max,epsilon,g,d_g)
    
    x_3 = 1
    
    
    #imprimindo os resultados:
    print('A raiz x^*_1 encontrada é %.17f ' % x_1[0])
    print('A tabela contendo as aproximacoes que encontramos ao longo das iteracoes é:')
    approx_printer(x_1[1], "grid")
    print()
    print('A raiz x^*_2 encontrada é %.17f ' % x_2[0])
    print('A tabela contendo as aproximacoes que encontramos ao longo das iteracoes é:')
    approx_printer(x_2[1], "grid")
    print()
    
    #Escrevendo em um arquivo .txt:

    f = open("lista_1_output.txt", "a")
    print('Tarefa 1, item 3', file = f)
    print('A raiz x^*_1 encontrada é %.17f ' % x_1[0], file = f)
    print('A tabela contendo as aproximacoes que encontramos ao longo das iteracoes é:', file = f)
    approx_printer_txt(x_1[1], f, "grid")
    print(file = f)
    print('A raiz x^*_2 encontrada é %.17f ' % x_2[0], file = f)
    print('A tabela contendo as aproximacoes que encontramos ao longo das iteracoes é:', file = f)
    approx_printer_txt(x_2[1], f, "grid")
    print(file = f)
    f.close()
    
    #####################Tarefa 1, item 4#############################################################
    
    #Calculando max f^(2)(x):
    set_max = [np.absolute(d2_f1(0)),np.absolute(d2_f1(x_1[0])),np.absolute(d2_f1(x_2[0])),np.absolute(d2_f1(x_3)),np.absolute(d2_f1(1))]
    max_d2_f = np.amax(set_max)
    
    
    #Calculando m_min:
    a = 0
    b = 1
    epsilon = 1e-8
    m_min = (b-a)*np.power(np.sqrt((12*epsilon)/((b-a)*max_d2_f)),-1)
    m_min = int(np.ceil(m_min))
    h_max = np.sqrt((12*epsilon)/((b-a)*max_d2_f))
    
    
    #Calculando T_m:
    t_m = trapezoidal_rule(0,1,m_min,f1)
    
    
    #Verificando se satisfaz a precisão:
    erro_int = np.power(b-a,3)/(12 * np.power(float(m_min),2)) * np.absolute(max_d2_f) 
    verif = np.absolute(t_m-1)
    
    #Imprimindo os resultados
    print("O valor maximo do módula da segunda derivada de f(x) e %.1f" % max_d2_f)
    print("O valor maximo do tamanho do intervalo e h_max = %.17f" % h_max)
    print("O valor minimo de subdivisoes do intervalo e m_min = %.f" % m_min)
    print("O valor de T_m utilizando m_min e %.17f" % t_m)
    print("O módulo do desvio do valor exato da intergral e",verif)
    print("O erro de integracao para funcoes suaves e", erro_int)
    print("T_m satisfaz a precião e o erro de integração?", verif <= erro_int and erro_int <= epsilon )
    print()
    
    #Imprimindo os resultados em um aqruivo .txt:

    f = open("lista_1_output.txt", "a")
    print('Tarefa 1, item 4', file = f)
    print("O valor maximo do módula da segunda derivada de f(x) e %.1f" % max_d2_f, file = f)
    print("O valor maximo do tamanho do intervalo e h_max = %.17f" % h_max, file = f)
    print("O valor minimo de subdivisoes do intervalo e m_min = %.f" % m_min, file = f)
    print("O valor de T_m utilizando m_min e %.17f" % t_m, file = f)
    print("O módulo do desvio do valor exato da intergral e",verif, file = f)
    print("O erro de integracao para funcoes suaves e", erro_int, file = f)
    print("T_m satisfaz a precião e o erro de integração?", verif <= erro_int and erro_int <= epsilon , file = f)
    print( file = f)
    f.close()
    
    #####################Tarefa 2, itens 1 e 2#############################################################
    
    #Dados
    a = 0
    b = 1
    n_max = 100000
    epsilon = 1e-10
    alpha_0 = 0.5
    
    #Método de Newton
    stn_time = time.time()
    raiz_Newton = newton_method(a,b,alpha_0,n_max,epsilon,fao,f1)
    newton_time = time.time() - stn_time
    
    #Método da bisseção
    stb_time = time.time()
    raiz_bisection = bisection_method(a,b,n_max,epsilon,fao)
    bisection_time = time.time() - stb_time
    
    #Imprimindo os resultados

    print('Tarefa 2, itens 1 e 2')
    print('A raiz obtida pelo metodo de Newton e tau = %.17f' % raiz_Newton[0])
    print("O tempo para gerar esta raiz utilizando o metodo de Newton foi: %s segundos" % (newton_time))
    print('As aproximacoes encontradas ao longo das iteracoes sao:')
    approx_printer(raiz_Newton[1], "grid")
    print()
    print('A raiz obtida pelo metodo de bissecao e tau = %.17f' % raiz_bisection[0])
    print("O tempo para gerar esta raiz utilizando o metodo de Newton foi: %s segundos" % (bisection_time))
    print('As aproximacoes encontradas ao longo das iteracoes sao:')
    approx_printer(raiz_bisection[1], "grid")
    print()


    #Imprimindo em um arquivo .txt
    f = open("lista_1_output.txt", "a")
    print('Tarefa 2, itens 1 e 2', file = f)
    print('A raiz obtida pelo metodo de Newton e tau = %.17f' % raiz_Newton[0], file = f)
    print("O tempo para gerar esta raiz utilizando o metodo de Newton foi: %s segundos" % (newton_time), file = f)
    print('As aproximacoes encontradas ao longo das iteracoes sao:', file = f)
    approx_printer_txt(raiz_Newton[1], f, "grid")
    print( file = f)
    print('A raiz obtida pelo metodo de bissecao e tau = %.17f' % raiz_bisection[0], file = f)
    print("O tempo para gerar esta raiz utilizando o metodo de Newton foi: %s segundos" % (bisection_time), file = f)
    print('As aproximacoes encontradas ao longo das iteracoes sao:', file = f)
    approx_printer_txt(raiz_bisection[1], f, "grid")
    print( file = f)
    f.close()
    
    #####################Tarefa 3, item 1#############################################################
    
    
    #Dados
    a = 0
    b= 1
    s = 5
    table_k = []
    #Laço percorrendo os valores de k desejados
    for k in range(3,10):
        
        #Subdivisões iniciais
        m_0 = s*np.power(2,k-1)
        #Criando a tabela para armazenar o resultado
        table_TE = [['x*h'],['T'],['E']]
        #Valor exato da integral para calcularmos o erro
        valor_integral = int_f_Tarefa_3(k,a,b)
        #Laço para calcular T(h), T(2h), ..., T(2^{k-1}h)
        for i in range(0,k):
            m = int(m_0/np.power(2,i))
            t_m = trapezoidal_rule_Tarefa_3(a,b,m,k,f_Tarefa_3)
            table_TE[0].append(np.power(2,i))
            table_TE[1].append(t_m)
            table_TE[2].append(np.absolute(t_m-valor_integral))
    
            #table_TE.append()
        table_k.append(table_TE)
        
        #Imprimindo os resultados

    print('Tarefa 3, item 1')
    for i,t in enumerate(table_k):
        print("Tabela para k = %.0f :" % (i+3))
        print(tabulate(t[1:], headers = t[0], tablefmt = "grid", numalign = "center", floatfmt = ".17f"))
        print()
        
        
    #Imprimindo em um arquivo .txt
    f = open("lista_1_output.txt", "a")
    print('Tarefa 3, item 1', file = f)
    for i,t in enumerate(table_k):
        print("Tabela para k = %.0f :" % (i+3), file = f)
        print(tabulate(t[1:], headers = t[0], tablefmt = "grid", numalign = "center", floatfmt = ".17f"), file = f)
        print(file = f)
    f.close()
        
        
        #####################Tarefa 3, item 2#############################################################
    
    
    
    #Laço sobre todos os valore de k
    x = []
    for k in range(3,10):
        #y = table_k[k-3][2][1:]
    #Gerando a matriz A:
        a = [np.ones(k, dtype = int)]
        for i in range(1,k):
            a.append([])
            for j in range(1,k+1):
                a[i].append((2**i)**(2*j))
                   
            
    #eliminação de Gauss
        x.append(eliminacaoGauss(a, table_k[k-3][2][1:], k))
            
    print('Tarefa 3, item 2')
    for k,i in enumerate(x):
        print("Resultado do sistema para k = %.0f:" % (k+3))
        header_result = ['2j'] 
        header_result = np.append(header_result,[2*j for j in range(1,k+4)])
        print(tabulate([np.append(['y_{2j}'],i)],headers = header_result, tablefmt = "grid", numalign = "center", floatfmt = ".17f"))
        print()
            
    #Imprimindo em um arquivo .txt
    f = open("lista_1_output.txt", "a")
    print('Tarefa 3, item 2', file = f)
    for k,i in enumerate(x):
        print("Resultado do sistema para k = %.0f:" % (k+3), file = f)
        header_result = ['2j'] 
        header_result = np.append(header_result,[2*j for j in range(1,k+4)])
        print(tabulate([np.append(['y_{2j}'],i)],headers = header_result, tablefmt = "grid", numalign = "center", floatfmt = ".17f"), file = f)
        print( file = f)
    f.close()
    
    #####################Tarefa 3, item 3#############################################################
    
    
    #Calculando B_4:
    table_b4 = [['k'],['B_4']]
    for k in range(3,10):
        table_b4[0].append(k)
        table_b4[1].append(b4(x[k-3][1],0,1,k,d3f_Tarefa_3))
            
    #Imprimindo os resultados
    print('Tarefa 3, item 3')
    print(tabulate([table_b4[1]], headers = table_b4[0], tablefmt = "grid", numalign = "center", floatfmt = ".17f"))
        
    #Imprimindo os resultados em um aqruivo .txt
    f = open("lista_1_output.txt", "a")
    print('Tarefa 3, item 3', file = f)
    print(tabulate([table_b4[1]], headers = table_b4[0], tablefmt = "grid", numalign = "center", floatfmt = ".17f"), file = f)
    f.close()
    
    
    
    
main()


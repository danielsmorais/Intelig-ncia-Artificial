%Autor: Daniel Silva de Morais
%Intelig�ncia Artificial - Engenharia de Computa��o - UFRN - NATAL/RN 
%Data: 17.03.2017 

%Fun��o LMS para Redes Neurais
%Recebe as amostras "x", os pesos iniciais "w", o resultado esperado "d", a
%taxa de apredizagem "h" e a precis�o requerida "e".
%OBS.: Os valores para "w" s�o interesantes serem aleat�rios

% Exemplo:
% x = [-1 -1 -1 -1 -1 -1 -1 -1 -1 1; -0.1 -1.6 -4.9 -3.2 -2.1 7.2 -4.2 -0.6 8.2 0.6; -5.9 -6.8 -9.7 -6.0 -4.5 9.6 6.8 0.5 -5.9 3.9]
% w = [1; 0.5; 1]
% d = [-1 -1 -1 -1 -1 1 1 1 1 1]
% h = 0.1
% e = 0.1

function [w, epoca] = lms(x,w,d,h,e)
    epoca = 0;                  %Contador de conjunto de amostras utilizados
    mse = 100;                  %Erro Quadr�tico M�dio - Inicializamos com um valor alto, para que seu valor n�o interfira na primeira itera��o
    [i j] = size(x);            %N�mero de linhas "i" e n�mero de Colunas "j"
    
    while(mse >= e)
        u = w'*x;               %Campo local induzido
        y = sign(u);            %Fun��o sinal
        erro = d - y;
        w = w + h*(x*(erro'));        
        mse = (sum((erro').^2))/j;        
        epoca = epoca + 1;
    end 
end
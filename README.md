## 02.2.2 - Normalização dos Dados
    A normalização é uma técnica usada para padronizar a escala dos dados numéricos na fase de pré-processamento de dados. No aprendizado de máquina, a normalização é importante porque os algoritmos geralmente são sensíveis à escala dos dados. Por exemplo, características com valores grandes podem pesar mais do que características com valores menores nos resultados finais, mesmo que as menores sejam mais significativas. O StandardScaler do Scikit-learn remove a média e escala os dados para uma variação unitária.

## 02.2.3 - Detecção e Tratamento de Outliers
    Outliers são observações que se desviam tanto das outras observações que levantam suspeitas sobre a sua validade. A presença de outliers pode levar a um modelo com desempenho pobre porque os métodos de aprendizado de máquina são sensíveis a eles. 
O método de Intervalo Interquartil (IQR) é uma técnica estatística usada para detectar outliers em um conjunto de dados. O IQR define os limites dentro dos quais a maioria dos dados de uma distribuição deve cair e é calculado como a diferença entre o terceiro quartil (Q3) e o primeiro quartil (Q1). Esses quartis são pontos de dados que dividem o conjunto de dados em quartos:

Q1 é o valor médio entre o menor número e a mediana do conjunto de dados; ele separa os 25% inferiores dos dados restantes.
Q3 é o valor médio entre a mediana e o maior número no conjunto de dados; ele separa os 25% superiores dos dados restantes.
O IQR reflete a faixa média e é menos afetado por extremos do que outras medidas, como a média.

Para identificar outliers, geralmente se define limites superior e inferior usando multiplicadores do IQR (comumente 1.5 vezes o IQR) adicionados a Q3 e subtraídos de Q1, respectivamente. Valores que caem fora desses limites são considerados atípicos ou outliers. Por exemplo:

Limite Inferior (LI) = Q1 - 1.5 * IQR
Limite Superior (LS) = Q3 + 1.5 * IQR
Valores abaixo do Limite Inferior ou acima do Limite Superior são marcados como outliers e podem ser removidos para evitar que distorçam a análise estatística ou os modelos de previsão.

## 02.2.4 - Detecção de Colinearidade
    A colinearidade ocorre quando duas ou mais características preditoras em um modelo de regressão são altamente correlacionadas, significando que uma pode ser linearmente prevista das outras com um grau substancial de precisão. Isso pode levar a coeficientes de regressão instáveis e dificuldade em interpretar quais características são realmente influentes na previsão. O Fator de Inflação da Variância (VIF) é uma medida que quantifica o aumento da variância de um coeficiente de regressão estimado devido à colinearidade.

### Resultados e Conclusões:

Na presente análise de multicolinearidade das variáveis envolvidas no modelo de predição de churn de clientes, foi observado que os indicadores de Inflação da Variância (VIF) apresentam valores que demonstram uma interdependência quase inexistente entre as variáveis estudadas. Em específico, a Pontuação de Crédito, com um VIF de 1.000275, sugere ausência de multicolinearidade, indicando que a variável é praticamente autônoma em relação às demais. Similarmente, a Idade, com um VIF de 1.001443, e o Tempo de Relacionamento com o banco, com um VIF de 1.000344, também se apresentam como independentes, sem inter-relações significativas que poderiam comprometer as análises.

Por outro lado, o Saldo em conta e o Número de Produtos, com VIFs de 1.102598 e 1.102901 respectivamente, ainda que apresentem valores ligeiramente acima de 1, estão bem aquém de qualquer patamar que denotaria preocupação com multicolinearidade. Tais achados apontam para uma leve correlação dessas variáveis com as outras no modelo, porém não a um grau que afete a precisão ou a validade das análises.

Diante desses resultados, pode-se inferir que não há indícios de multicolinearidade que possam prejudicar a confiabilidade do modelo proposto. A manutenção de todas as variáveis selecionadas se justifica e é aconselhável, tendo em vista que cada uma contribui com informações distintas e valiosas para a predição de churn, sem que haja uma sobreposição que poderia levar a uma inflação artificial na significância estatística dos coeficientes. Portanto, os indicadores utilizados são candidatos robustos para a aplicação em modelos de regressão múltipla e outros métodos estatísticos que pressupõem a independência entre os preditores.


## 02.3.1 - Visualização de Dados (Vazão de Contas)

O gráfico apresenta duas categorias distintas: "Retido" e "Saiu", indicando a fidelidade dos clientes à empresa. Observamos que 79,6% dos clientes foram retidos, enquanto 20,4% dos clientes saíram ou cessaram o relacionamento com a empresa no período analisado. Este gráfico é fundamental para estabelecer um ponto de partida para a nossa análise, destacando a importância de compreender os fatores que contribuem para a vazão de clientes.

A seguir, discutiremos o significado deste churn e como ele pode impactar a operação e a sustentabilidade financeira da empresa. A identificação dos padrões e tendências subjacentes aos dados de churn não só nos ajudará a entender o comportamento do cliente, mas também servirá como alicerce para a construção de modelos preditivos. Estes modelos têm como objetivo prever a probabilidade de churn de clientes individuais, permitindo que a empresa tome ações preventivas e estratégicas para melhorar a retenção.

O gráfico nos informa, visualmente, que há uma proporção significativa de churn que precisa ser endereçada. Além disso, indica a necessidade de investigar as características dos clientes que estão mais propensos a sair, o que será explorado em análises subsequentes. A interpretação deste gráfico sugere que é necessário um balanceamento nos dados.

## 02.3.1 - Visualização de Dados (Vazão de Contas)
    Em muitos conjuntos de dados utilizados para tarefas de classificação, é comum encontrar uma distribuição desigual entre as classes alvo. Esse desbalanceamento pode prejudicar o desempenho de modelos de aprendizado de máquina, levando-os a favorecer a classe majoritária e negligenciar a minoritária. A classe minoritária é frequentemente de maior interesse e, por isso, métodos de balanceamento de classes são empregados para mitigar esse problema e melhorar a capacidade de generalização do modelo.
    Nesse caso, estamos utilizando o SMOTETomek une as forças do SMOTE e do Tomek Links para criar um conjunto de dados balanceado que não só equilibra as classes através da geração de dados sintéticos, mas também refina a fronteira de decisão ao eliminar os Tomek Links. Esta abordagem combinada é particularmente útil para lidar com conjuntos de dados onde a classe minoritária é crucial para a análise e não pode ser negligenciada.

# 03 - Desenvolvimento do Modelo

### 03.1 - Seleção do Modelo

#### __Random Forests (Florestas Aleatórias)__:

* __Por que escolher__: São robustas contra overfitting devido ao seu mecanismo de ensemble. Além disso, podem capturar relações não-lineares entre as variáveis sem a necessidade de escalar ou normalizar os dados. Também são úteis por fornecerem importâncias das variáveis, o que pode ser valioso para interpretação e insights de negócios.
* __Quando usar__: Random Forest é uma boa escolha quando você precisa de um modelo que possa lidar com interações complexas entre variáveis e é desejável ter uma ideia de quais variáveis são as mais importantes para a previsão.

#### __Support Vector Machines (SVM)__:

* __Por que escolher__: SVM é eficaz em espaços de alta dimensão e em casos onde o número de dimensões é maior que o número de amostras. É versátil graças aos diferentes kernels que podem ser especificados para a decisão de fronteira. Além disso, é eficiente em encontrar margens máximas de separação entre classes, o que pode ser muito útil para problemas de churn que tendem a ter margens de classificação não tão claras.
* __Quando usar__: SVM é uma escolha adequada quando a separação de classe não é clara, e é necessário um modelo capaz de criar fronteiras de decisão complexas e não-lineares. É particularmente útil quando os dados não são linearmente separáveis. No entanto, o SVM pode não ser ideal para conjuntos de dados muito grandes devido ao seu maior tempo de treinamento em comparação com outros modelos como Random Forest ou XGBoost, e pode ser desafiador escolher e ajustar o kernel correto. Use SVM quando o equilíbrio entre a complexidade do modelo e a capacidade de generalização for uma prioridade, e haja tempo e recursos para uma cuidadosa otimização de hiperparâmetros.

#### __XGBoost__:

* __Por que escolher__: É uma implementação otimizada de Gradient Boosting que é famosa por ganhar várias competições de Machine Learning. Oferece desempenho rápido, tratamento eficaz de valores ausentes, e a capacidade de regularizar para evitar overfitting.
* __Quando usar__: XGBoost é adequado quando se procura performance superior em um modelo que também pode ser regularizado para evitar overfitting. Além disso, é uma boa escolha se o tempo de treinamento e os recursos computacionais são uma consideração, pois é mais eficiente que a implementação padrão do GBM.

# 04 - Resultados e Conclusões

## 04.1 Síntese dos Resultados
O presente trabalho teve como objetivo desenvolver modelos preditivos para a previsão de churn em um conjunto de dados de clientes. Foram aplicados três modelos distintos: Random Forest, Support Vector Machine (SVM) e XGBoost. Os resultados obtidos foram avaliados com base nas métricas de precisão, recall e f1-score.

* __Random Forest__: O modelo apresentou uma precisão geral (accuracy) de 86%, com um desempenho destacado na classificação da classe majoritária (clientes retidos), alcançando uma precisão de 87% e um recall de 96%. Para a classe minoritária (clientes que saíram), a precisão foi de 74% com um recall de 43%. O f1-score geral para este modelo foi de 0.84, indicando uma boa capacidade preditiva.

* __SVM__: O modelo SVM mostrou uma precisão geral de 80%, com uma precisão de 81% para a classe majoritária e um recall elevado de 98%. No entanto, para a classe minoritária, a precisão caiu para 31% e o recall para 3%. Isso resultou em um f1-score geral de 0.72, apontando para uma performance inferior, especialmente na detecção da classe de interesse (clientes que saíram).

* __XGBoost__: O modelo XGBoost apresentou uma precisão geral de 85%, com uma precisão de 88% para a classe majoritária e um recall de 95%. A classe minoritária teve uma precisão de 68% e um recall de 46%. O f1-score geral foi de 0.84, semelhante ao Random Forest, o que denota uma boa performance geral.

## 04.2 Conclusões
Os modelos testados ofereceram insights valiosos sobre os fatores associados à probabilidade de churn dos clientes. O modelo Random Forest provou ser o mais eficaz, equilibrando precisão e capacidade de detecção de churn. Enquanto isso, o SVM não conseguiu capturar eficientemente a classe minoritária, o que é crítico para a aplicação em questão. O XGBoost, por sua vez, apresentou resultados competitivos, sendo apenas ligeiramente inferior ao Random Forest.

A escolha do Random Forest como o modelo mais eficiente é devido à sua capacidade de manejar a complexidade do conjunto de dados e fornecer uma boa generalização. O XGBoost também se mostrou uma opção robusta, com a vantagem adicional de facilitar a interpretação dos fatores que influenciam a previsão.

## 04.3 Sugestões para Trabalhos Futuros
Para estudos futuros, recomenda-se a exploração de métodos de balanceamento de dados mais avançados para melhorar a detecção da classe minoritária. Ademais, a implementação de técnicas de seleção de características pode ser benéfica para melhorar a interpretabilidade dos modelos e possivelmente sua performance. Por fim, é sugerido o teste de outros algoritmos de aprendizado de máquina e a aplicação de técnicas de otimização de hiperparâmetros para aprimorar ainda mais a precisão dos modelos preditivos.

Este estudo contribui para a literatura de previsão de churn, demonstrando a aplicabilidade e a eficácia de modelos de machine learning nesse contexto. As conclusões aqui apresentadas podem ser utilizadas para informar estratégias de retenção de clientes e impulsionar a tomada de decisão baseada em dados em ambientes corporativos.

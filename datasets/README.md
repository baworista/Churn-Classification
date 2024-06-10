# Identyfikacja klientów o wysokim ryzyku odejścia do konkurencji (CHURN)

Jesteś analitykiem firmy telekomunikacyjnej. Twoim zadaniem jest opracowanie modelu identyfikującego klientów o wysokim ryzyku odejścia do konkurencji (CHURN) oraz ocena efektywności ekonomicznej jego wdrożenia.

Aktualna sytuacja jest następująca:

1. Na jednym kliencie mamy 700 USD marży.
2. Klientowi, któremu kończy się niebawem umowa, oferujemy 100 USD dolarów zachęty (bonus), by z nami został. Nie wykorzystujemy w tym celu żadnego modelu.
3. Koszt nawiązania takiego kontaktu (praca telemarketingu) wynosi 50 USD
4. Nie każdy z klientów, do którego zadzwonimy, decyduje się na przedłużenie umowy: w takiej sytuacji ponosimy koszt 50 USD, nie wydajemy jednak 100 USD na bonus.

Z działu sprzedaży otrzymałeś plik z następującymi danymi:

**State**: the US state in which the customer resides, indicated by a two-letter abbreviation; for example, OH or NJ

**Area Code**: the three-digit area code of the corresponding customer’s phone number

**Phone**: the remaining seven-digit phone number

**Account Length**: the number of days that this account has been active

**Int’l Plan**: whether the customer has an international calling plan

**VMail Plan**: whether the customer has a voice mail feature

**VMail Message**: presumably the average number of voice mail messages per month

**Day Mins**: the total number of calling minutes used during the day

**Day Calls**: the total number of calls placed during the day

**Day Charge**: the billed cost of daytime calls

**Eve Mins, Eve Calls, Eve Charge**: the billed time, # of calls and cost for calls placed during the evening

**Night Mins, Night Calls, Night Charge**: the billed time, # of calls and cost for calls placed during nighttime

**Intl Mins, Intl Calls, Intl Charge**: the billed time, # of calls and cost for international calls

**CustServ Calls**: the number of calls placed to Customer Service

**Churn?**: whether the customer left. 0: stayed (no churn), 1: left (churn)

Powodzenia!

## Zadania

1. Analiza opisowa i model bazowy 

   - [ ] Przejrzyj dane ("Analyze")
   - [ ] Stwórz model bazowy: Lab > ML Prediction > Churn
   - [ ] Zidentyfikuj zwycięzcę i zapisz wartość ROC AUC

2. Model 1.0: pierwsza inżynieria cech 

   - [ ] Stwórz kolumnę "Total_mins" zawierającą sumę liczby minut rozmów dziennych, wieczornych, nocnych i międzynarodowych

   - [ ] Stwórz kolumnę "Total_charge" zawierającą sumę kosztów dziennych, wieczornych, nocnych i międzynarodowych

   - [ ] Stwórz model 1.0, uwzględniający nowe zmienne

   - [ ] Zidentyfikuj zwycięzcę, zapisz wartość ROC AUC

     ***Jaki był wpływ nowych zmiennych na efektywność modelu?***

3. Model 2.0: uwzględnienie segmentacji 

   - [ ] Stwórz model segmentacji (ML Clustering). Wyłącz "phone" i "churn"!

   - [ ] Stwórz model 2.0, uwzględniający nowe zmienne (klastry)

   - [ ] Zidentyfikuj zwycięzcę, zapisz wartość ROC AUC

     ***Jaki był segmentacji na efektywność modelu?***

5. Model 3.0: inżynieria cech, nowe algorytmy 

   - [ ] Stwórz model klasyfikacji, wyłączając z analizy różne zmienne

   - [ ] Zidentyfikuj zwycięzcę, zapisz wartość wybranej miary oceniającej jakość modelu

   - [ ] Stwórz model klasyfikacji, próbując nowych algorytmów

   - [ ] Zidentyfikuj zwycięzcę, zapisz wartość ROC AUC

     ***Jaki był wpływ nowych wyłączania nowych zmiennych i wprowadzania nowych algorytmów na efektywność modelu?***

6. **Efektywność ekonomiczna najlepszego modelu** 

   - [ ] Dla najlepszego modelu, wpisz parametry Performance > Confusion Matrix opierając się na założeniach przedstawionych w zadaniu
   - [ ] Oblicz zysk na jednym rekordzie
   
7. Model 4.0: wybór i optymalizacja najlepszej miary oceny jakości modelu

   1. Wybierz Twoim zdaniem najlepszą miarę oceny jakości modelu

   2. Zoptymalizuj trenowanie modelu pod kątem nowej miary

   3. Oblicz zysk na jednym rekordzie

      ***W jaki sposób wybór Twojej nowej miary wpłynął na zysk na 1 rekordzie?***

###### 

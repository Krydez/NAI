# Podsumowanie: Wpływ funkcji jądra (Kernel Functions) na wyniki klasyfikacji SVM

Support Vector Machine (SVM) polega na znalezieniu optymalnej hiperpłaszczyzny rozdzielającej klasy. Funkcje jądra (Kernels) pozwalają algorytmowi przenieść dane do wyższego wymiaru, co umożliwia rozdzielenie danych, które nie są liniowo separowalne w pierwotnej przestrzeni.

Poniżej przedstawiono analizę wpływu różnych typów kerneli na podstawie przeprowadzonych eksperymentów (zbiory *Banknote Authentication* oraz *Pima Diabetes*).

## 1. Kernel Liniowy (`kernel='linear'`)
* **Działanie:** Szuka prostej linii (lub płaszczyzny) rozdzielającej dane.
* **Wyniki:**
    * Bardzo wysoka skuteczność (bliska 98-99%) dla zbioru **Banknote**, ponieważ dane te są naturalnie dobrze odseparowane.
    * Niższa skuteczność dla zbiorów złożonych (jak **Diabetes**), gdzie relacje między zmiennymi nie są proste.
* **Zastosowanie:** Najlepszy dla dużych zbiorów danych o wielu cechach (np. klasyfikacja tekstu), gdzie dane są często liniowo separowalne. Jest najszybszy obliczeniowo.

## 2. Kernel RBF (`kernel='rbf'`) - Radial Basis Function
* **Działanie:** Tworzy nieliniowe granice decyzyjne oparte na odległościach (przypominające okręgi lub "wyspy"). Jest to domyślny kernel w bibliotece scikit-learn.
* **Wyniki:**
    * Osiągnął **100% skuteczności** dla zbioru **Banknote**.
    * Zazwyczaj daje najwyższe wyniki dla **Diabetes**, ponieważ potrafi dopasować się do nieregularnych skupisk danych medycznych.
* **Parametry (`gamma`, `C`):**
    * Duże `gamma` prowadzi do bardzo skomplikowanych granic (ryzyko *overfittingu*).
    * Małe `gamma` tworzy gładsze, bardziej ogólne granice.

## 3. Kernel Wielomianowy (`kernel='poly'`, `degree=3`)
* **Działanie:** Odwzorowuje dane w oparciu o wielomiany zadanego stopnia.
* **Wyniki:** Potrafi modelować krzywizny granic decyzyjnych. Dla zbioru Banknote wynik był zbliżony do RBF.
* **Ryzyko:** Przy wysokim stopniu wielomianu (`degree > 3`) czas obliczeń drastycznie rośnie, a model ma tendencję do "przeuczenia" (zbyt mocnego dopasowania do danych treningowych).

## 4. Kernel Sigmoidalny (`kernel='sigmoid'`)
* **Działanie:** Działa podobnie do sieci neuronowej (funkcja aktywacji).
* **Wyniki:** W przeprowadzonych testach osiągnął **najgorszy wynik** (często w okolicach 75% lub mniej dla Banknote, co jest wynikiem słabym w porównaniu do 100% RBF).
* **Wniosek:** Kernel ten jest specyficzny i rzadko działa dobrze "z pudełka". Wymaga bardzo precyzyjnego dostrojenia parametrów i specyficznych danych.

## Wnioski końcowe
Wybór funkcji jądra ma **kluczowy wpływ** na jakość klasyfikacji:
1. Dla danych prostych i wyraźnie oddzielonych (Banknote) nawet prosty model **liniowy** jest wystarczający.
2. Dla danych zaszumionych i złożonych (Medyczne/Diabetes) **RBF** jest zazwyczaj najlepszym wyborem startowym.
3. Źle dobrany kernel (np. Sigmoid bez tuningu) może drastycznie obniżyć skuteczność modelu, nawet jeśli dane są dobrej jakości.
# Guide de présentation : ce qu'il faut comprendre pour défendre le projet

Ce document est pour toi, pas pour le jury. Il couvre trois choses : les concepts de finance
derrière chaque onglet, ce que le code fait concrètement, et les questions qu'on va te poser.
Objectif : que chaque chiffre affiché soit un chiffre que tu peux expliquer en deux phrases.


> **Démo en ligne (phase 2).** La « morning sheet » lit le dashboard à ta place et se régénère chaque
> jour ouvré après la clôture US : [https://morning-sheet-olive.vercel.app/vol](https://morning-sheet-olive.vercel.app/vol). Les deux projets frères sont sur le même site
> ([https://morning-sheet-olive.vercel.app/dispersion](https://morning-sheet-olive.vercel.app/dispersion), [https://morning-sheet-olive.vercel.app/basket](https://morning-sheet-olive.vercel.app/basket)). Le code est dans
> `src/vol_dashboard/sheet.py` ; chaque chiffre vient d'une fonction existante de `src/`. En entretien,
> ouvre la sheet d'abord, l'app Streamlit ensuite pour les détails.

---

## 1. Les concepts, onglet par onglet

### 1.1 Vol réalisée vs vol implicite

- **Vol réalisée (RV)** : ce qui s'est passé. Écart-type des rendements log quotidiens,
  annualisé en multipliant par √252 (252 jours de bourse). Exprimée en % : 20 veut dire
  20 % par an.
- **Vol implicite (IV)** : ce que le marché des options price pour le futur. On l'obtient en
  inversant un modèle de pricing (Black-Scholes ou Black-76) à partir du prix d'une option.
- **Règle du pouce à connaître** : vol annuelle / 16 ≈ mouvement quotidien attendu.
  VIX à 16 = le marché attend environ 1 % par jour sur le S&P (16 / √252 ≈ 1).
- Les indices CBOE (VIX, VXN, OVX, GVZ) sont des IV à 30 jours. Ce sont eux qu'on compare à
  la RV sur 21 jours de bourse (≈ 30 jours calendaires).

### 1.2 Pourquoi quatre estimateurs de vol réalisée

Un seul prix de clôture par jour, c'est peu d'information. Les estimateurs de range
exploitent le haut et le bas de la journée.

| Estimateur | Utilise | Force | Faiblesse |
|---|---|---|---|
| Close-to-close | clôtures | simple, capte les gaps overnight | bruité |
| Parkinson (1980) | high, low | environ 5 fois plus efficace | aveugle aux gaps overnight |
| Garman-Klass (1980) | open, high, low, close | encore plus efficace | suppose pas de drift ni de gap |
| Yang-Zhang (2000) | tout, découpé en overnight + intraday | robuste aux gaps et au drift | plus de paramètres |

Ce qu'il faut savoir dire : "quand Parkinson sort en dessous du close-to-close, c'est que les
mouvements se font overnight, pas en séance". C'est de l'information, pas un bug.

### 1.3 Le vol risk premium (VRP)

- Définition dans le projet : **VRP = IV aujourd'hui − RV des 21 jours suivants**.
- Il est positif en moyenne (3,5 points sur le S&P depuis 2015, positif 83 % des jours).
  Le vendeur d'options est payé pour porter le risque de variance, comme un assureur.
- Il devient massivement négatif en crise : −70 points en février 2020. L'assureur paie le
  sinistre.
- Pourquoi il existe : les investisseurs veulent de la protection, les vendeurs sont peu
  nombreux et exigent une prime. C'est un prix du risque, pas une inefficience.
- Attention au look-ahead : on compare l'IV de la date t à la RV de t+1 à t+21. La série
  s'arrête donc 21 jours avant la dernière date.

### 1.4 Le vol cone

Pour chaque fenêtre (10, 21, 42, 63, 126, 252 jours), on calcule la RV glissante sur cinq ans
et on regarde sa distribution (min, 10e, 25e, médiane, 75e, 90e, max). Le cône est large aux
fenêtres courtes parce qu'elles sont bruitées. Lecture : si la RV courante est en haut du
cône et que l'IV est en dessous, le marché anticipe un retour à la moyenne.

### 1.5 La term structure du VIX

- VIX9D, VIX, VIX3M, VIX6M, VIX1Y : IV à 9, 30, 91, 182, 365 jours.
- **Contango** (normal) : la courbe monte avec la maturité. Plus d'incertitude à long terme,
  plus de prime de risque.
- **Backwardation** (stress) : la courbe s'inverse, le court terme dépasse le long terme. Le
  marché price un choc maintenant et un retour au calme ensuite.
- Le ratio VIX / VIX3M au-dessus de 1 est le signal d'inversion. Inversé 7 % des jours
  depuis 2015.
- **VVIX** : la vol implicite du VIX lui-même (options sur VIX). **SKEW** : indice CBOE du
  prix des queues de distribution, 100 = lognormale, plus c'est haut plus les puts loin de
  la monnaie sont chers.

### 1.6 Le skew et le smile

- Le smile : l'IV n'est pas la même à tous les strikes. Black-Scholes supposerait une IV
  plate. La réalité est une courbe.
- **Equity : put skew.** Les puts hors de la monnaie sont plus chers que les calls. Raisons :
  demande de protection contre les krachs, effet de levier (une baisse augmente
  l'endettement relatif), corrélation négative entre prix et vol.
- **Commodities : souvent call skew.** Sur le pétrole ou le gaz, le choc redouté est un choc
  d'offre qui fait monter le prix. Aujourd'hui USO a un risk reversal à +7,5 alors que SPY
  est à −4,8.
- Les mesures desk :
  - **Risk reversal 25 delta** = IV(call 25d) − IV(put 25d). Signe négatif = put skew.
  - **Butterfly 25 delta** = moyenne des deux ailes − ATM. Mesure la convexité du smile,
    donc le prix des queues.
  - **Skew 90/110** = IV(put 90 %) − IV(call 110 %), version en moneyness.
- Pourquoi parler en delta et pas en strike : le delta normalise par la vol et la maturité.
  Un put 25 delta est "aussi loin" de la monnaie sur le S&P à 15 de vol que sur le pétrole
  à 50 de vol. Les traders quotent en delta.

### 1.7 Forward, parité call-put et Black-76

- **Parité call-put** : C − P = DF × (F − K), avec DF le facteur d'actualisation et F le
  forward. Elle ne dépend d'aucun modèle, c'est de l'arbitrage pur.
- Donc avec un call et un put au même strike, on **déduit le forward** : F = K + (C − P) / DF.
  C'est ce que fait le code sur les strikes proches de la monnaie, et il prend la médiane.
- Pourquoi c'est important : le forward intègre dividendes (SPY) et coût de portage (ETF
  commodities). Si on prend le spot à la place, les IV des puts et des calls divergent. C'est
  exactement l'erreur dans le champ IV de Yahoo.
- **Black-76** = Black-Scholes écrit sur le forward au lieu du spot. Même formule, mais F
  remplace S e^{(r−q)T}. Le delta forward est N(d1) pour un call, N(d1) − 1 pour un put.
- Taux flat à 4 % : sur un à trois mois, DF ≈ 0,99 ; une erreur de 1 % sur le taux déplace
  une IV de 20 % de moins de 0,02 point. Pas besoin d'une courbe.

### 1.8 Le VIX est un variance swap, pas une moyenne d'IV

- Un **variance swap** paie la variance réalisée contre un strike fixé à l'avance. On peut
  le répliquer avec un portefeuille d'options OTM pondérées en 1/K² (Demeterfi, Derman, Kamal,
  Zou, 1999). L'intuition : cette pondération donne une exposition à la variance qui ne
  dépend pas du niveau du spot.
- Le CBOE applique exactement cette formule à deux expiries encadrant 30 jours puis interpole
  en variance totale :
  σ² = (2/T) Σ (ΔK_i / K_i²) e^{RT} Q(K_i) − (1/T)(F/K0 − 1)²
  avec Q(K) le mid de l'option OTM au strike K, K0 le premier strike sous le forward.
- Le projet reproduit ça sur SPY : 17,65 contre 17,79 officiel. L'écart vient de SPY contre
  SPX (dividendes, cotations retardées, expiries un peu différentes) et du fait que l'officiel
  est la clôture de la veille.
- Chiffre à retenir : **les puts portent 73 % de la variance** alors qu'ils sont la moitié
  des strikes. C'est le skew qui est "dans" le VIX. Si le skew monte, le VIX monte même
  sans changement de l'ATM.

### 1.9 Prévision de vol : qui prédit le mieux ?

Question : à 21 jours, le marché des options prédit-il mieux la vol future que les modèles
basés sur le passé ?

Les modèles :
- **RV22** : la vol reste là où elle est. La référence naïve.
- **HAR-RV (Corsi 2009)** : régression de log RV future sur log RV d'hier (Parkinson sur un
  jour), de la semaine (5 j) et du mois (22 j). Trois horizons qui approximent la mémoire
  longue de la vol. Simple, robuste, utilisé sur les desks.
- **GARCH(1,1)** : σ²_t = ω + α ε²_{t−1} + β σ²_{t−1}. La variance d'aujourd'hui dépend du
  choc d'hier et de la variance d'hier. α + β proche de 1 = forte persistance. Estimé par
  maximum de vraisemblance, puis on somme les variances prévues sur 21 jours.
- **Implicite** : le VIX tel quel comme prévision.
- **HAR + IV** : HAR avec log VIX comme régresseur supplémentaire. Si le coefficient du VIX
  "absorbe" les autres, l'implicite contient toute l'information. C'est un test
  d'encompassing.

La méthode, et pourquoi elle est propre :
- **Hors échantillon** : à chaque date, les modèles ne sont ajustés que sur les données
  connues à cette date. Fenêtre croissante, refit tous les 21 jours, départ après deux ans.
  Sans ça, les résultats sont du look-ahead et ne valent rien.
- **RMSE** en points de vol : simple mais récompense les prévisions qui se cachent vers la
  moyenne.
- **QLIKE** (Patton 2011) : la perte robuste quand la vol réalisée est elle-même un proxy
  bruité. C'est celle que la littérature utilise.
- **Mincer-Zarnowitz** : régression RV = α + β × prévision. Prévision non biaisée si
  (α, β) = (0, 1). β < 1 signifie qu'un point de prévision vaut moins d'un point de réalisé.

Le résultat sur le S&P depuis 2017 :

| Modèle | Biais | RMSE | QLIKE | R² MZ |
|---|---|---|---|---|
| RV22 | 0,0 | 10,6 | 0,73 | 0,24 |
| HAR-RV | −2,0 | 9,2 | 0,66 | 0,28 |
| GARCH(1,1) | +1,6 | 11,0 | 0,60 | 0,10 |
| Implicite | +3,6 | 9,2 | 0,44 | 0,35 |
| HAR + IV | −1,1 | 8,5 | 0,56 | 0,34 |

La phrase à dire : "l'implicite est le meilleur prédicteur en information (R², QLIKE) mais il
est biaisé de 3,6 points vers le haut, c'est le variance risk premium. Le combiner avec
l'historique corrige le biais et donne la meilleure RMSE. C'est exactement ce qu'un desk
fait : partir de l'implicite et le débiaiser."

Limite à annoncer soi-même : les cibles se chevauchent (horizon 21 jours, une observation
par jour), donc les écarts de quelques centièmes ne sont pas significatifs.

### 1.10 Pétrole : courbe, Samuelson, régimes

- **Courbe des futures** : prix par mois de livraison. **Contango** = les échéances
  lointaines sont plus chères (coût de stockage, marché bien approvisionné).
  **Backwardation** = le front est plus cher (pénurie physique, convenience yield élevé).
- **Roll yield** : ce qu'un long futures gagne ou perd en roulant d'un contrat à l'autre.
  Positif en backwardation (on rachète moins cher), négatif en contango.
- **Effet Samuelson (1965)** : la vol d'un contrat augmente à l'approche de son expiration.
  L'information de court terme (stocks hebdomadaires, OPEP, météo) frappe le front ; l'arrière
  de la courbe est ancré par le coût marginal de production. Mesuré aujourd'hui : 32 % de vol
  sur le contrat à 1 mois, 11 % à 12 mois.
- **Spread Dec/Dec** (CLZ26 contre CLZ27) : la jauge classique de la forme de courbe chez
  les traders pétrole, et la seule reconstruisable sur données gratuites puisque les deux
  pattes ont des années d'historique.
- **Résultat honnête** : la corrélation spread / OVX est nulle (0,04) parce que 2020 a mis
  la courbe en super-contango avec une vol à plus de 100. Le lien "backwardation = vol haute"
  ne tient pas sur cet échantillon. En revanche le **premium** est trié : vendre de la vol
  pétrole a rapporté +4,3 points en backwardation contre −0,6 en contango. Dire ça devant un
  jury vaut plus qu'un graphe trop beau : tu montres que tu testes au lieu de confirmer.

---

## 2. Ce que fait le code

### 2.1 Architecture

```
app.py                  interface Streamlit, onglets, cache
src/vol_dashboard/
  universe.py           liste des actifs (ticker, indice de vol, ETF pour les options, racine futures)
  data.py               téléchargement yfinance, miroir CSV, chaîne d'options, courbe futures
  metrics.py            estimateurs de RV, cône, VRP, term structure
  options.py            mid, forward implicite, inversion Black-76, delta, risk reversal
  vix.py                strip CBOE, réplication du VIX
  forecast.py           HAR, GARCH, course hors échantillon, pertes
  crude.py              contrats individuels, panel Samuelson, spread Dec/Dec, régimes
  charts.py             figures Plotly
scripts/snapshot_skew.py   capture quotidienne du skew
tests/                  13 tests pytest
```

Principe : la logique financière est dans `src/`, testable sans interface. `app.py` ne fait
que charger, appeler et afficher.

### 2.2 Le flux de données

1. `load_ohlc(ticker)` appelle yfinance, garde Open/High/Low/Close, écrit un CSV dans
   `data/`. Avec `offline=True` il relit le CSV. Le format est volontairement celui d'un
   export Bloomberg (Date, O, H, L, C) pour pouvoir brancher un vrai extract.
2. `option_chain(ticker, target_days)` récupère les expiries listées, choisit la plus proche
   de la cible (au moins 7 jours pour éviter le bruit de l'échéance), renvoie calls et puts
   avec bid, ask, dernier prix, volume, open interest.
3. `futures_curve(root, exchange)` construit les codes de contrats (CL + lettre de mois +
   année + .NYM, lettres FGHJKMNQUVXZ pour janvier à décembre) et prend le dernier prix de
   chacun.

### 2.3 Les calculs qu'il faut pouvoir expliquer

**Rendements et vol glissante** (`metrics.py`)

```python
r = np.log(close / close.shift(1))
rv = r.rolling(21).std() * np.sqrt(252) * 100
```

`shift(1)` décale d'un jour, `rolling(21).std()` calcule l'écart-type sur une fenêtre
glissante. La vol forward utilise `shift(-21)` : la valeur à la date t devient la RV calculée
sur t+1 à t+21. C'est ce qui permet le VRP sans look-ahead.

**Inversion de Black-76** (`options.py`)

```python
brentq(lambda s: black76_price(F, K, t, s, is_call, df) - price, 1e-4, 5.0)
```

`brentq` est un solveur de racine : il cherche la vol σ telle que le prix modèle égale le
prix de marché. Borné entre 0,01 % et 500 %. Si le prix est sous la valeur intrinsèque,
pas de solution, on renvoie NaN plutôt qu'une valeur fausse.

**Forward implicite**

```python
fwd = K + (C_mid - P_mid) / DF   # sur chaque strike à ±3 % du spot
F = median(fwd)
```

Médiane et non moyenne : un strike avec une cotation stale ne pollue pas le résultat.

**Interpolation sur le delta**

```python
np.interp(-0.25, puts["delta"], puts["iv_pct"])
```

Il n'y a jamais exactement un put à 25 delta coté, donc on interpole linéairement entre les
deux voisins. Si la cible est hors de la plage cotée (ailes trop fines), on renvoie NaN :
c'est le cas de UNG à 90 jours.

**Le strip CBOE** (`vix.py`)

1. Trouver le strike où |C − P| est minimal, en déduire F.
2. K0 = premier strike sous F.
3. Partir de K0 vers le bas pour les puts, vers le haut pour les calls ; s'arrêter après deux
   bids nuls consécutifs (règle CBOE).
4. ΔK_i = moitié de l'écart entre les strikes voisins.
5. Sommer ΔK/K² × e^{RT} × mid, appliquer la formule, faire les deux expiries, interpoler en
   variance totale sur 30 jours, racine, ×100.

**La boucle hors échantillon** (`forecast.py`)

```python
for i in range(min_train, len(feats)):
    if (i - min_train) % refit == 0:
        beta = ols(X[:i - 21], y[:i - 21])   # refit sur ce qui est connu à la date i
    forecast[i] = exp(beta @ x[i])
```

Le détail qui compte : l'entraînement s'arrête à i − 21 parce que la cible de la ligne i − 20
utilise des rendements postérieurs à i. Oublier ça, c'est du look-ahead déguisé.

`_ols` est une régression par moindres carrés avec `np.linalg.lstsq`. GARCH vient de la
librairie `arch` : on l'ajuste sur les rendements en %, on prévoit 21 variances
quotidiennes, on prend leur moyenne annualisée.

**Panel Samuelson** (`crude.py`)

Pour chaque contrat vivant : rendements, RV 21 jours, temps restant jusqu'à l'expiration en
mois. Empiler les 30 contrats (`pd.concat`), découper en buckets de maturité (`pd.cut`),
médiane par bucket (`groupby`). Les jours à volume nul sont retirés : un prix de règlement
figé sur un contrat 2029 en 2019 ressemble à une vol nulle et fausserait l'effet.

Date d'expiration CL : trois jours ouvrés avant le 25 du mois précédant le mois de
livraison. Codée avec `pd.offsets.BDay`.

### 2.4 Streamlit, en deux phrases

Un script Python ré-exécuté à chaque interaction. `@st.cache_data(ttl=3600)` évite de
retélécharger : la fonction n'est rappelée que si ses arguments changent ou si l'heure est
passée. Les onglets, colonnes et métriques sont des appels de fonction, les graphes sont des
objets Plotly passés à `st.plotly_chart`.

### 2.5 Les tests

- **Estimateurs** : on simule un mouvement brownien géométrique à 25 % de vol avec 390 pas
  intraday par jour, on construit des OHLC, et chaque estimateur doit retrouver 25 ± 1,5. Le
  ratio des variances close-to-close / Parkinson doit être entre 3 et 8 (Parkinson annonce 5).
- **Options** : aller-retour prix → IV → prix ; sur une chaîne synthétique dont le forward est
  à 101,5 pour un spot à 100, le forward est retrouvé à 0,05 près ; un smile synthétique avec
  put skew donne un risk reversal négatif, avec call skew positif.
- **VIX** : sur une chaîne Black-Scholes à vol plate de 12, 20 ou 45 %, le strip rend la
  variance d'entrée à 0,2 point près, avec et sans taux ; une chaîne skewée met plus de poids
  sur les puts.

Pourquoi c'est important devant un jury : les tests vérifient la finance (le code retrouve
les valeurs théoriques), pas seulement que le code tourne.

### 2.6 Le snapshot quotidien

`scripts/snapshot_skew.py` fait pour chaque actif et pour 30 et 90 jours : chaîne → nettoyage
→ métriques de skew → une ligne dans `data/skew_history.csv`, dédoublonnée sur (date, proxy,
tenor). Le workflow GitHub Actions le lance à 22h15 UTC en semaine et commite. L'argument :
personne ne donne l'historique du risk reversal 25 delta gratuitement, donc le projet le
construit lui-même.

---

## 3. Les questions probables et les réponses courtes

**Pourquoi l'implicite est-elle supérieure à la réalisée en moyenne ?**
Parce que vendre des options c'est vendre de l'assurance. La prime compense le risque de
sinistre (2020 : −70 points). C'est un prix du risque, documenté depuis Bakshi et Kapadia
2003 / Carr et Wu 2009.

**Pourquoi le skew equity est-il négatif ?**
Demande structurelle de puts (fonds, assureurs), effet de levier (la baisse augmente la vol),
mémoire de 1987. Sur les commodities c'est souvent l'inverse parce que le risque redouté est
une hausse (choc d'offre).

**Pourquoi Black-76 et pas Black-Scholes ?**
Parce que tout est écrit sur le forward, qui absorbe dividendes et portage sans avoir à les
estimer. Sur un ETF avec dividendes, Black-Scholes sur le spot fausse les IV.

**Les options sur ETF sont américaines, Black-76 est européen. Problème ?**
Oui, petit. Pour des OTM à un ou trois mois, la prime d'exercice anticipé est négligeable
sauf pour les puts très dans la monnaie, qu'on n'utilise pas (OTM seulement). Je le signale
comme limite.

**Le VIX, c'est quoi exactement ?**
La racine du strike d'un variance swap à 30 jours sur le S&P, calculé à partir d'un strip
d'options OTM pondérées en 1/K². Pas une moyenne d'IV. Je l'ai reconstruit à 0,15 point
près depuis SPY.

**Pourquoi les puts pèsent 73 % du strip ?**
Parce qu'ils sont plus chers (skew) et que la pondération 1/K² donne plus de poids aux
strikes bas. Le VIX est donc très sensible au skew, pas seulement à l'ATM.

**Qu'est-ce que QLIKE et pourquoi pas juste la RMSE ?**
Une fonction de perte qui reste cohérente quand la cible (la RV) est un proxy bruité de la
vraie variance. La RMSE favorise les prévisions trop lisses. Patton 2011.

**Vos prévisions ont des cibles qui se chevauchent, ça pose quoi ?**
Les erreurs sont autocorrélées, donc les écarts-types classiques des régressions MZ sont
faux. Je ne les reporte pas et je ne commente que les écarts larges. Une version propre
sous-échantillonnerait tous les 21 jours ou utiliserait Newey-West.

**Pourquoi le VIX/VIX3M s'inverse en stress ?**
Parce qu'un choc est pricé comme temporaire : vol haute maintenant, retour à la moyenne
ensuite. L'inversion dure rarement plus de quelques semaines.

**Qu'est-ce que l'effet Samuelson et pourquoi existe-t-il ?**
La vol d'un futures monte à l'approche de l'échéance. Les infos de court terme (stocks,
OPEP) touchent le front ; le long terme est ancré par le coût marginal de production, qui
bouge peu.

**Pourquoi la corrélation courbe / OVX est nulle alors que la théorie dit l'inverse ?**
Parce que 2020 domine l'échantillon : super-contango et vol à 100 en même temps. Sur un
échantillon sans 2020 la relation est positive. Je préfère montrer le résultat tel quel.

**Qu'est-ce que vous feriez avec Bloomberg ?**
Remplacer les CSV par SPX Index, VIX Index, CL1 Comdty, OVX Index, prendre VSTOXX et RVX
qui manquent, prendre la surface de vol historique (pour un vrai historique de skew sans
attendre les snapshots), et des cotations non retardées pour le strip.

**Combien de temps ça a pris, et qu'est-ce qui a été dur ?**
Dire la vérité : la partie dure a été la donnée. Découvrir que l'IV de Yahoo était fausse,
reconstruire le forward, gérer les contrats expirés qui disparaissent, les jours sans volume
sur les contrats longs. Le code de pricing est court une fois qu'on sait ce qu'on veut.

**Vous avez utilisé une IA ?**
Oui, comme outil, et le code est à moi dans le sens où je peux expliquer chaque ligne et
chaque choix. Ce qui compte c'est le journal de décisions dans le README et les tests qui
vérifient la finance. Ne jamais prétendre le contraire, les jurys le sentent.

---

## 4. Le pitch de deux minutes

"J'ai construit un dashboard de volatilité sur indices actions et matières premières, en
données gratuites, qui compare vol réalisée et implicite et répond à trois questions.

Première question : est-ce que je comprends ce qu'est la vol implicite ? J'ai reconstruit le
VIX depuis les options SPY avec la méthodologie CBOE, le strip pondéré en 1/K², et je tombe
à 0,15 point du print officiel. Au passage, j'ai découvert que l'IV fournie par Yahoo viole
la parité call-put, donc je recalcule les IV avec un forward implicite et Black-76.

Deuxième question : est-ce que l'implicite prédit mieux la vol future que l'historique ?
Course hors échantillon entre HAR-RV, GARCH, le VIX et un modèle combiné, évaluée en RMSE,
QLIKE et Mincer-Zarnowitz. Réponse : l'implicite a le plus d'information mais un biais de
3,6 points, le variance risk premium, et le modèle combiné qui corrige ce biais gagne.

Troisième question, sur mon angle commodities : comment la vol du pétrole se comporte-t-elle
le long de la courbe ? L'effet Samuelson est net, 32 % de vol sur le contrat à un mois contre
11 % à douze mois. Et le régime de courbe trie le premium : vendre de la vol pétrole rapporte
en backwardation, perd en contango.

Le tout est testé, 13 tests qui vérifient que les estimateurs retrouvent la vol d'une
simulation et que le strip CBOE rend la variance théorique, et un pipeline quotidien construit
un historique de skew que personne ne donne gratuitement."

---

## 5. Les limites à annoncer avant qu'on te les sorte

- Données retardées et gratuites : SPY au lieu de SPX, ETF américains évalués avec un modèle
  européen, taux flat.
- Pas d'historique de surface de vol : le skew historique commence au premier snapshot.
- Cibles de prévision chevauchantes, pas d'inférence statistique sur les écarts fins.
- Les contrats longs sont peu liquides : leur vol est sous-estimée dans le panel Samuelson,
  d'où le filtre sur le volume.
- Le VRP est calculé en points de vol, pas en P&L d'une vraie position : une stratégie
  réelle a des coûts de transaction, du delta-hedge, des marges.

Annoncer ces limites soi-même, c'est la différence entre quelqu'un qui a compris et quelqu'un
qui a assemblé.

---

## 6. L'onglet Harvesting vol (ajouté ensuite)

### Le straddle delta-hedgé

- On vend un straddle ATM à un mois au niveau du VIX, on couvre le delta chaque soir avec
  Black-Scholes à cette même vol, on tient jusqu'à l'échéance. P&L = prime − payoff + P&L de
  couverture − coûts.
- **L'identité gamma-theta** : le P&L d'une option couverte vaut ½ Σ Γ S² (σ²impl dt − r²). Le
  vendeur encaisse l'écart entre variance implicite et variance réalisée, pondéré par le gamma.
  Le code le vérifie : 98 % de corrélation entre le P&L simulé et la formule.
- **Path dependence** : à écart implicite-réalisée égal, le P&L varie selon l'endroit où le spot
  se trouvait quand le gamma était grand. Un variance swap n'a pas ce problème, d'où le nuage
  comparatif.
- Chiffres S&P depuis 2005 : 261 trades, 79 % gagnants, 0,6 % du spot par mois après coûts,
  pire mois septembre 2008. On couvre à la vol implicite parce que c'est la seule
  implémentable : on ne connaît pas la réalisée à l'avance.

### Forward vol et FOMC

- Forward variance entre deux tenors : σ²f = (T2 σ2² − T1 σ1²)/(T2 − T1). Avec VIX9D et VIX
  on obtient la vol attendue entre le jour 9 et le jour 30.
- Si un FOMC tombe dans les 9 premiers jours, le VIX9D porte sa variance en plus de la
  forward. L'excès donne le mouvement implicite : √(9/365 × (σ9D² − σf²)).
- Résultat : sur 94 réunions, 0,69 % pricé contre 0,80 % réalisé. **La vol d'événement n'est
  pas surpayée, la vol générique l'est.** Le carry est dans les jours sans nouvelle.
- Live : avec les deux expiries SPY autour de l'événement, V1 = (n1 − 1) b + (b + e) et
  V2 = (n2 − 1) b + (b + e), donc b = (V2 − V1)/(n2 − n1) et e = V1 − n1 b. La base b peut
  aussi venir de l'ATM 30 jours, plus propre quand d'autres événements polluent la deuxième
  expiry.

### Le signal

- Écart = implicite − prévision HAR+IV. Plus l'écart est grand, plus le premium futur est
  grand (3 à 6 points), mais ne vendre que sur grand écart perd plus de carry qu'il n'évite de
  pertes. **Résultat négatif assumé** : l'écart dit combien on est payé, pas quand ça craque.

**Question jury** : "Pourquoi ne pas couvrir à la vol réalisée ?" Parce qu'elle n'est connue
qu'après. Couvrir à l'implicite donne un P&L path-dependent mais implémentable ; couvrir à la
réalisée donnerait un P&L lisse mais fictif.

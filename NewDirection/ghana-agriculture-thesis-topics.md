# Thesis Topic Options — Agriculture & Horticulture in Ghana

**Prepared:** 14 August 2026
**Constraint set:** must build *and apply* a model (CANDO); master's timeframe; R/Python and Stata available; primary data access uncertain.

---

## How this document is built

Every topic below is engineered so that **the core model can be estimated on secondary data alone**. Where a small primary survey would materially strengthen the work, it is listed as an *upgrade path*, never as a dependency. This is deliberate: the most common way an otherwise good thesis dies is a gatekeeper who never returns the dataset. Nothing here has a single point of failure.

Feasibility ratings weigh four things equally: data obtainability, methodological complexity, time to completion, and how convincingly the topic satisfies "build a model" rather than "describe a situation."

---

## Part 1 — The Ghanaian data landscape, tiered by access risk

Before the topics, the honest picture of what you can actually get. Read this first; it explains most of the feasibility scores.

### Tier A — Free, public, downloadable today (zero gatekeeper risk)

| Source | What it gives you | Coverage | Format |
|---|---|---|---|
| **UN Comtrade** | Bilateral trade flows by HS code, value + quantity | 1990s–present, annual & monthly | API / CSV |
| **FAOSTAT** | Ghana crop area, production, yield, trade, food balance | 1961–present, annual, national | CSV / API |
| **EU RASFF Portal** | Every border rejection & alert on Ghanaian food: product, hazard, date, notifying country, action taken | 1979–present, notification-level | Web portal, scrapable |
| **EUROPHYT / TRACES-NT** | Plant-health interceptions (harmful organisms) by origin & commodity | Annual reports | PDF / dashboards |
| **CHIRPS** | Gridded rainfall, 0.05° resolution | 1981–present, dekadal/monthly | GeoTIFF, Earth Engine |
| **ERA5 / ERA5-Land** | Gridded temperature, evapotranspiration, soil moisture | 1950–present, hourly→monthly | NetCDF, CDS API |
| **TerraClimate / SPEI database** | Drought indices (PDSI, SPEI at multiple scales) | 1958–present, monthly | NetCDF |
| **MODIS NDVI/EVI** | Vegetation vigour, 250m–1km | 2000–present, 16-day | Earth Engine |
| **WFP VAM / FAO GIEWS-FPMA** | Monthly market prices, Ghanaian markets (Accra, Kumasi, Tamale, Techiman) for maize, rice, cassava, yam, sorghum, millet, groundnut | ~2000s–present | CSV / API |
| **APHLIS+** | Modelled post-harvest loss estimates for cereals & legumes, sub-national | 2003–present | CSV / web |
| **World Bank WDI, WITS, UNCTADstat** | Macro controls, tariffs, NTM counts, RCA inputs | Long | API |
| **CEPII Gravity database** | Distance, contiguity, language, colonial ties, RTAs — everything a gravity model needs | 1948–present | CSV |

**This tier alone is enough to complete Topics 1, 2, 4, 5, 6, 8, 9 and 12.** That is the single most important sentence in this document.

### Tier B — Obtainable with a formal request and some patience (moderate risk)

- **MoFA / SRID — *Agriculture in Ghana: Facts and Figures***. Regional area harvested, production and yield for ~12–15 major crops. Published annually as PDF; older editions circulate freely. Expect to extract tables with `camelot`/`tabula` in Python. **Caveat that matters:** Ghana went from 10 to 16 administrative regions in December 2018, so any panel spanning that date must be harmonised back to the original 10 regions or the panel breaks.
- **GSS microdata** — GLSS6 (2012/13), GLSS7 (2016/17), Ghana Census of Agriculture 2017/18, AHIES. Agriculture modules give plot-level input use, output, and sales. Request via the GSS microdata catalogue; academic requests are routinely granted but can take weeks.
- **Ghana Socioeconomic Panel Study (GSPS)** — ISSER/Yale, three waves. A genuine household panel with agriculture modules. Best available route to *panel* farm-level efficiency without collecting anything yourself.
- **Ghana Commodity Exchange (GCX)** — price and volume series for maize, soya, sorghum, sesame, paddy rice since 2018. Short series, but real and under-analysed.
- **GEPA — Non-Traditional Export statistics.** Annual NTE value/volume by product and destination. The annual reports are the accessible form; disaggregated series usually need a formal request.
- **Ghana Meteorological Agency (GMet)** — station data, typically fee-based. Prefer CHIRPS/ERA5 unless you specifically need station-level validation.

### Tier C — Depends on a relationship, and may not arrive (high risk — never build a thesis's core on this)

- **FAGE / SPEG / PAMPEAG / VEPEAG member-level export records** (exporter-level volumes, destinations, rejections).
- **MoFA PPRSD** phytosanitary certificate and interception logs.
- **CSIR institute trial data** (CRI, SARI, FRI, OPRI) — varietal trials, storage trials, loss experiments.
- **COCOBOD** farm-level and purchasing-clerk data.
- **TCDA** cashew/shea/mango registries.

Treat everything in Tier C as a **bonus chapter**, not a foundation. If it arrives, it turns a good thesis into an excellent one. If it doesn't, nothing is lost.

---

## Part 2 — The twelve topics

---

### Topic 1 — Predicting border-rejection risk for Ghanaian agri-food exports

**1. Proposed title**
*Modelling Sanitary and Phytosanitary Border-Rejection Risk for Ghanaian Agri-Food Exports to the European Union: A Count-Panel and Machine-Learning Approach, 2000–2025*

**2. Research problem / gap**
Ghana's horticultural export ambitions are repeatedly undercut not by production capacity but by market-access failure at the border. The most visible episode was the self-imposed suspension of five vegetable lines to the EU from January 2015 to end-2017, triggered by a run of interceptions for harmful organisms. Despite this, there is no published model that quantifies *which* Ghanaian products, hazards and seasons carry elevated rejection risk, or how that risk moves with export volume and regulatory change. Existing work on African food-safety rejections is overwhelmingly cross-country and descriptive; Ghana-specific, hazard-disaggregated risk modelling does not exist.

**3. Research questions**
1. What is the incidence and hazard composition of EU border rejections of Ghanaian agri-food exports, per unit of trade exposure, 2000–2025?
2. Which product, hazard, seasonal and exposure characteristics predict rejection counts and rejection *rates*?
3. Did the 2015–2017 suspension and subsequent compliance reforms produce a measurable, sustained reduction in rejection intensity relative to comparator West African exporters?
4. Can a classifier assign a usable ex-ante risk score to a product–destination–month consignment profile?

**4. Variables**
- **Dependent:** (a) count of rejection notifications per product–year; (b) rejection *rate* = notifications ÷ export volume (exposure-adjusted); (c) binary hazard class (mycotoxin / pesticide residue / microbiological / plant-health / composition-labelling) for the classification stage.
- **Independent:** export volume and value (exposure offset), product HS group, hazard type, month/season, harvest-season rainfall and humidity in the sourcing region (aflatoxin is climate-driven), EU regulatory tightening events (MRL revisions, emergency measures), a post-2018 reform indicator, destination member state, comparator-country fixed effects.

**5. Model to build**
Two-stage.
- *Stage 1 — inferential:* Poisson / negative binomial panel with an exposure offset (`log` trade volume), country and product fixed effects; zero-inflated variant tested because many product–years legitimately have zero notifications. Overdispersion test decides NB vs Poisson.
- *Stage 2 — predictive:* gradient-boosted classifier (XGBoost / LightGBM) predicting rejection occurrence and hazard class, evaluated with time-based cross-validation (train on pre-2020, test 2020–2025), reported as AUC-PR rather than accuracy given class imbalance. **SHAP** values used to expose which features drive risk, so the model is interpretable to a regulator, not a black box.
- *Optional third stage:* difference-in-differences or synthetic control around the 2015 suspension, using Côte d'Ivoire, Nigeria, Kenya and Cameroon as donors.

**6. Data required**
RASFF notification-level extract for Ghana and 4–6 comparator countries; UN Comtrade export volumes at HS-4/HS-6 matched to RASFF product categories; CHIRPS/ERA5 climate for sourcing regions; a hand-coded timeline of EU regulatory events.

**7. Ghanaian / external sources**
RASFF Portal (primary, free); EUROPHYT-TRACES annual reports; UN Comtrade; GEPA NTE reports for corroboration of Ghanaian export composition; MoFA PPRSD phytosanitary logs (Tier C bonus); FAGE for exporter-side context interviews.

**8. Unit of analysis**
Primary: product (HS-4) × year. Secondary: individual rejection notification. Comparative: country × product × year.

**9. Methodology**
Scrape/export RASFF into a notification-level dataset → harmonise RASFF product text to HS codes via a documented concordance (this concordance is itself a contribution) → merge Comtrade exposure → construct exposure-adjusted rates → estimate count panel → estimate ML classifier with temporal validation → SHAP interpretation → DiD/synthetic control on the 2015 episode → translate the fitted model into a simple risk-scoring table for GEPA/PPRSD.

**10. Expected contribution**
*Academic:* first exposure-adjusted, hazard-disaggregated rejection model for Ghana; a reusable RASFF→HS concordance; evidence on whether a trade suspension actually improved compliance or merely suppressed volume. *Practical:* a risk score that lets PPRSD target scarce inspection capacity at high-risk product–season combinations rather than inspecting uniformly.

**11. Feasibility: 9/10**
All core data is free, public and downloadable this week. No institutional permission required for the core model. Method matches both halves of your toolkit.

**12. Challenges / limitations**
- **The main risk:** Ghana-only notification counts may be thin in some product–years. *Mitigation, built in from the start:* pool 5–7 West/East African exporters and treat Ghana as the focal country with country fixed effects. This raises N and strengthens the comparative claim.
- RASFF product descriptions are free text and need careful, documented harmonisation to HS.
- Rejections measure *detected* non-compliance, not true non-compliance; detection intensity varies by member state. Address explicitly with destination fixed effects and a limitations discussion.

---

### Topic 2 — Climate–yield response and regional yield forecasting

**1. Proposed title**
*Rainfall Variability, Heat Stress and Crop Yield in Ghana: A Panel and Machine-Learning Model of Climate–Yield Response Across Agro-Ecological Zones, 1992–2024*

**2. Research problem / gap**
Ghana's rainfed cropping is exposed to shifting onset dates, intra-season dry spells and rising heat, but national planning still leans on rules of thumb about "good" and "bad" rainfall years. Total seasonal rainfall is a poor predictor of yield; *distribution* within the season is what matters. Few Ghanaian studies build a yield-response model that uses agronomically meaningful weather constructs (onset, dry-spell length, growing degree days, days above a critical temperature) rather than seasonal totals, and fewer still test whether responses differ by agro-ecological zone.

**3. Research questions**
1. How do agronomically constructed rainfall and temperature variables affect yields of Ghana's major crops, and do total-rainfall specifications understate the true effect?
2. Do climate–yield elasticities differ systematically across the Coastal Savannah, Forest, Transition, Guinea Savannah and Sudan Savannah zones?
3. Is there evidence of non-linear heat damage above a temperature threshold, and where is that threshold?
4. Can a model trained on historical data forecast regional yields ahead of harvest with useful accuracy, and does ML beat the econometric specification out of sample?

**4. Variables**
- **Dependent:** yield (mt/ha) by region × crop × year, for maize, rice, cassava, yam, groundnut, sorghum, millet, cowpea, soybean, plantain.
- **Independent:** growing-season total rainfall and its square; rainfall onset date and cessation; number and maximum length of dry spells (>7, >10, >14 consecutive dry days); growing degree days; degree days above 30 °C / 32 °C; SPEI at 3- and 6-month scales; NDVI at mid-season; area harvested; fertilizer availability proxy; region and year fixed effects; a Planting for Food and Jobs (2017+) policy indicator.

**5. Model to build**
- *Core:* two-way fixed-effects panel yield-response model (region and year FE, region-specific linear trends), with piecewise-linear or spline temperature terms to identify a damage threshold; standard errors clustered by region and Conley-corrected for spatial correlation.
- *Comparison:* random forest and XGBoost trained on the same features, evaluated by leave-one-year-out and blocked (forward-chaining) cross-validation. **SHAP** used to compare what the ML model learned against the econometric coefficients — where they agree, confidence rises; where they diverge, that divergence is a finding.
- *Application:* a within-season forecasting exercise using only weather observed up to a cut-off date (e.g. end of the second month of the season) to produce a pre-harvest yield forecast, scored against a naive climatological benchmark and a random-walk benchmark.

**6. Data required**
Regional crop area/production/yield panel, 1992–2024; gridded daily rainfall and temperature aggregated to region and to the crop growing window; NDVI; agro-ecological zone shapefile.

**7. Ghanaian / external sources**
MoFA SRID *Agriculture in Ghana: Facts and Figures* (regional yield panel — the backbone); FAOSTAT (national-level fallback and cross-check); CHIRPS + ERA5-Land (climate, free); MODIS NDVI; GMet station data for validation if affordable; CSIR-CRI and CSIR-SARI for agronomic parameters (planting windows, critical temperature thresholds, varietal maturity periods) — these can come from published CSIR literature, not necessarily unpublished data.

**8. Unit of analysis**
Region × crop × year (≈10 harmonised regions × 10 crops × 30 years ≈ 3,000 observations — comfortably enough).

**9. Methodology**
Digitise SRID tables → harmonise 16 regions back to 10 → define crop-specific growing windows per zone using CSIR agronomic literature → extract zonal statistics from CHIRPS/ERA5 in Python (`rioxarray`, `geopandas`, or Earth Engine) → construct weather features → estimate FE panel → estimate ML models with blocked CV → SHAP comparison → within-season forecast evaluation → project yields under simple warming scenarios (+1 °C, +2 °C) holding the estimated response fixed.

**10. Expected contribution**
*Academic:* first Ghana-wide, zone-differentiated, agronomically-specified climate–yield panel with an ML benchmark; identification of a heat-damage threshold. *Practical:* a pre-harvest regional yield forecast that MoFA and the National Food Buffer Stock Company could use for procurement and early warning.

**11. Feasibility: 9/10**
Climate data is free and unlimited. The only real work is digitising SRID PDFs, which is a bounded one-to-two week task with the right Python tooling. FAOSTAT is a working fallback if regional digitising proves harder than expected.

**12. Challenges / limitations**
- Region boundary change in 2018 must be handled explicitly; document the harmonisation.
- SRID production statistics are themselves estimates with known measurement error; acknowledge and, where possible, cross-check against FAOSTAT.
- Aggregating gridded weather to administrative regions introduces aggregation bias; weight grid cells by cropland mask (from a land-cover product) rather than by area, and say so.
- Fertilizer and input data at regional-year level are patchy; you may need a proxy.

---

### Topic 3 — Technical efficiency of Ghanaian smallholder farms

**1. Proposed title**
*Technical Efficiency and Its Determinants Among Ghanaian Smallholder Crop Farmers: A Stochastic Metafrontier Analysis Across Agro-Ecological Zones*

**2. Research problem / gap**
Ghana's yield gaps are usually attributed to input access, but a large share may be inefficiency in converting inputs already held. Most Ghanaian efficiency studies use small, single-district convenience samples, which cannot support national inference and cannot compare zones on a common technology. A metafrontier framework — which separates *managerial* inefficiency from *technology gap* — has rarely been applied to nationally representative Ghanaian data.

**3. Research questions**
1. What is the mean technical efficiency of Ghanaian smallholder crop farms, nationally and by zone?
2. How much of the observed yield gap is managerial inefficiency versus a technology gap between zones?
3. Which household and institutional factors — extension contact, credit access, education, land tenure security, irrigation, gender of the plot manager, cooperative membership — reduce inefficiency?
4. What is the estimated output gain from closing the efficiency gap, and how does that compare with the gain from raising input use?

**4. Variables**
- **Dependent:** value of crop output per farm (or per plot).
- **Frontier inputs:** cultivated area, family and hired labour days, fertilizer quantity/value, seed, agrochemicals, capital/equipment services.
- **Inefficiency determinants:** age, education, household size, extension visits, credit received, off-farm income, distance to market, irrigation, tenure type, gender, cooperative/FBO membership, zone.

**5. Model to build**
Battese–Coelli (1995) stochastic frontier with a simultaneously estimated inefficiency-effects model (translog functional form, tested against Cobb–Douglas by likelihood ratio), estimated separately by agro-ecological zone, then combined into a **stochastic metafrontier** to decompose zone efficiency into managerial efficiency and technology gap ratio. Cross-check with a bootstrapped DEA (Simar–Wilson two-stage) for robustness. Optionally add a sample-selection-corrected frontier (Greene 2010) if adopters/non-adopters of a technology are compared.

**6. Data required**
Nationally representative farm-household data with plot-level inputs and outputs, plus household characteristics.

**7. Ghanaian / external sources**
GSS — GLSS7 (2016/17) agriculture module; Ghana Census of Agriculture 2017/18; GLSS6 for a repeated cross-section. ISSER/Yale **Ghana Socioeconomic Panel Study** if you want a true panel (enabling true fixed-effects frontiers and efficiency *change*). MoFA extension records for context.

**8. Unit of analysis**
Farm household (or plot, if the survey supports plot-level input allocation).

**9. Methodology**
Request and clean GSS microdata → construct consistent input/output aggregates with price deflation → test functional form → estimate zone-specific frontiers in Stata (`sfcross`/`sfpanel`) or R (`frontier`, `sfaR`) → estimate metafrontier → bootstrap technology gap ratios → regress efficiency determinants within the one-step framework → counterfactual output simulation.

**10. Expected contribution**
*Academic:* nationally representative, zone-decomposed efficiency estimates for Ghana with a metafrontier separation that most prior work omits. *Practical:* tells MoFA whether extension money or input subsidy money buys more output at the margin — a live policy argument around Planting for Food and Jobs.

**11. Feasibility: 8/10**
Method is well-established with mature software in both your toolkits. The score is 8 rather than 9 only because it depends on GSS microdata access, which is normally granted to students but is a queue you don't control. Start that request in week one.

**12. Challenges / limitations**
- Cross-sectional frontiers cannot separate inefficiency from unobserved heterogeneity; be explicit, and prefer GSPS panel data if you can get it.
- Mixed-crop farms complicate output aggregation; use value aggregation with documented deflators.
- Efficiency scores are sensitive to functional form and distributional assumptions on the inefficiency term — run half-normal, truncated-normal and exponential and report all three.
- Some inefficiency determinants (credit, extension) are plausibly endogenous; acknowledge, and consider control-function approaches.

---

### Topic 4 — Regional agricultural productivity growth: DEA and Malmquist decomposition

**1. Proposed title**
*Decomposing Agricultural Productivity Growth in Ghana's Regions, 1995–2024: A Data Envelopment Analysis and Malmquist Total Factor Productivity Approach*

**2. Research problem / gap**
Ghana has run successive input-subsidy and mechanisation programmes on the premise that output growth is input-constrained. Whether the country's agricultural growth has come from *more inputs*, *better use of inputs*, or *technical progress* has not been decomposed at regional level over a long horizon. Without that decomposition, policy cannot tell whether it is buying growth or buying inputs.

**3. Research questions**
1. What is the level and trend of technical efficiency across Ghana's regions in crop production?
2. How does total factor productivity change decompose into efficiency change (catch-up) and technical change (frontier shift)?
3. Which regions define the efficient frontier, and what is the measured slack for the rest?
4. Do productivity trends shift around major policy interventions (fertilizer subsidy 2008, Planting for Food and Jobs 2017)?

**4. Variables**
- **Outputs:** aggregate crop production value (or multiple crop outputs in a multi-output DEA).
- **Inputs:** area cultivated, agricultural labour force, fertilizer consumption, tractor/mechanisation proxy, irrigated area.
- **Second-stage explanatory:** rainfall, road density, extension officer ratio, credit penetration, literacy.

**5. Model to build**
Input- and output-oriented **DEA** under CRS and VRS (scale efficiency as the ratio), computed year by year; **Malmquist TFP index** decomposed into efficiency change × technical change, further split into pure efficiency and scale efficiency change; **bootstrapped DEA (Simar–Wilson)** for confidence intervals on scores; **second-stage truncated regression** of efficiency scores on environmental variables with bootstrap correction. Optionally a Färe–Primont or Hicks–Moorsteen index for a properly transitive comparison across time.

**6. Data required**
Regional panel of crop output and input use, 1995–2024, plus environmental covariates.

**7. Ghanaian / external sources**
MoFA SRID (output, area); GSS Population & Housing Census + GLSS (agricultural labour force by region); MoFA fertilizer distribution and mechanisation service centre records; GIDA (irrigated area); CHIRPS (rainfall); Ghana Highway Authority / GSS (road density).

**8. Unit of analysis**
Region × year (10 harmonised regions × 30 years = 300 DMU-years — adequate for DEA, which is not data-hungry).

**9. Methodology**
Assemble regional input–output panel → run DEA per year in R (`Benchmarking`, `deaR`) or Stata → compute Malmquist indices between consecutive years → bootstrap → decompose → second-stage truncated regression → interpret against policy timeline.

**10. Expected contribution**
*Academic:* a long-horizon, decomposed regional TFP series for Ghana that does not currently exist in the public literature. *Practical:* a direct answer to whether subsidy-era growth was frontier-shifting or merely input-adding.

**11. Feasibility: 8/10**
Data overlaps almost entirely with Topic 2, so the two share a build cost. DEA is unambiguously "a model" and is computationally light. Marked down slightly because assembling consistent regional *labour* and *fertilizer* series across 30 years is genuinely fiddly.

**12. Challenges / limitations**
- DEA is deterministic — measurement error in SRID data propagates directly into scores. Bootstrapping mitigates but does not remove this; consider a parallel stochastic frontier as a robustness check.
- Only 10 DMUs per year is a small frontier; the curse of dimensionality means you must keep the input set tight (3–4 inputs maximum) or almost everything scores efficient.
- Regional labour force must be interpolated between census years; document the interpolation.

---

### Topic 5 — Gravity model of horticultural exports and untapped market potential

**1. Proposed title**
*Determinants and Untapped Potential of Ghana's Horticultural Export Flows: A Structural Gravity Model with Sanitary Stringency and AfCFTA Effects*

**2. Research problem / gap**
Ghana's export strategy targets aggressive non-traditional export growth, but target markets are typically chosen on narrative grounds rather than estimated potential. No study estimates a properly specified structural gravity model for Ghana's *horticultural* exports specifically, incorporating standards stringency, and then uses the fitted model to compute market-by-market export potential — the gap between predicted and actual trade.

**3. Research questions**
1. What determines the value of Ghana's horticultural exports across destinations and products?
2. How much do destination SPS/TBT stringency and tariff structures suppress flows, relative to distance and market size?
3. Which destination–product pairs show the largest gap between gravity-predicted potential and actual exports?
4. Is there a detectable AfCFTA-period effect on intra-African horticultural flows from Ghana?

**4. Variables**
- **Dependent:** bilateral export value (USD) by product × destination × year, including zeros.
- **Independent:** destination GDP and GDP per capita, distance, contiguity, common official language, colonial history, RTA membership, applied tariff, count of SPS/TBT notifications affecting the product in the destination, exchange rate, landlocked status, AfCFTA entry-into-force indicator, exporter–product and destination–year fixed effects.

**5. Model to build**
**Poisson Pseudo-Maximum Likelihood (PPML) structural gravity** with high-dimensional fixed effects (`ppmlhdfe` in Stata; `fixest::fepois` in R), retaining zero flows rather than dropping them — this is the methodological core and where most naive gravity work goes wrong. Robustness: Heckman two-step for the extensive/intensive margin split, and a two-part model. Then **export potential** computed as fitted-minus-actual, ranked, with bootstrap uncertainty bands. Optionally a machine-learning stage (random forest on the same covariates) to check for interactions the log-linear structure misses.

**6. Data required**
Bilateral horticultural export flows by HS code and destination, 2000–2024; standard gravity covariates; tariff and NTM data.

**7. Ghanaian / external sources**
UN Comtrade / WITS (flows, tariffs); **GEPA NTE reports** (Ghanaian-source validation of product coverage and destination composition — an important cross-check that most gravity papers on Ghana skip); CEPII gravity database (all geography/history covariates); WTO I-TIP and UNCTAD TRAINS (SPS/TBT notifications); World Bank WDI (macro).

**8. Unit of analysis**
Product (HS-4/HS-6) × destination × year.

**9. Methodology**
Define the horticultural HS basket (pineapple, banana, mango, papaya, vegetables, cashew, yam, and processed derivatives) → build the full rectangular panel including zeros → merge gravity covariates → estimate PPML with fixed effects → robustness across specifications → compute and rank export potential → validate the top-ranked opportunities against GEPA/FAGE strategy documents and discuss why gaps persist.

**10. Expected contribution**
*Academic:* a correctly specified PPML gravity model for Ghanaian horticulture with standards variables — a real gap in the literature. *Practical:* an evidence-ranked target list of destination–product pairs for GEPA's market development budget, replacing narrative targeting.

**11. Feasibility: 8/10**
Data is entirely Tier A and the observation count is in the tens of thousands, so statistical power is never the issue. Software is mature. Marked at 8 because gravity modelling is methodologically well-trodden, so originality has to come from the horticultural focus, the standards variables and the potential-estimation application — you must actively build that, not assume it.

**12. Challenges / limitations**
- Mirror-data discrepancies between Ghana's reported exports and partners' reported imports; use partner-reported imports as the preferred series and justify it.
- SPS stringency measured by notification counts is a crude proxy for real regulatory burden; discuss.
- AfCFTA implementation is uneven and recent, so any AfCFTA effect will be imprecisely estimated — frame as exploratory, not causal.

---

### Topic 6 — Forecasting horticultural export earnings

**1. Proposed title**
*Forecasting Ghana's Horticultural Export Earnings: A Comparative Evaluation of Econometric, Machine-Learning and Hybrid Models*

**2. Research problem / gap**
GEPA sets multi-year non-traditional export targets, and the Bank of Ghana's foreign-exchange planning depends on export receipts, yet no published work benchmarks forecasting methods on Ghana's horticultural export series. Whether classical seasonal time-series methods or machine learning perform better on these particular series — short, seasonal, structurally broken — is an open and directly useful question.

**3. Research questions**
1. What are the trend, seasonality and structural-break characteristics of Ghana's major horticultural export series?
2. Which forecasting class — SARIMA/SARIMAX, ETS, TBATS, Prophet, gradient boosting, LSTM, or a hybrid — minimises out-of-sample error at 3-, 6- and 12-month horizons?
3. Do exogenous predictors (exchange rate, world commodity prices, EU import demand, rainfall in the sourcing zone) improve forecasts beyond univariate structure?
4. Are the differences in forecast accuracy statistically significant?

**4. Variables**
- **Dependent:** monthly export value and volume for pineapple, banana, mango, papaya, cashew, yam and fresh vegetables.
- **Independent (exogenous):** GHS/USD and GHS/EUR exchange rates, world price indices for the relevant commodity, EU/destination industrial production or import demand index, sourcing-zone rainfall, fuel/freight cost index, month dummies, break indicators.

**5. Model to build**
A full forecasting **tournament** with a fixed protocol: rolling-origin (walk-forward) evaluation, identical train/test splits across all models, error measured by RMSE, MAE and MASE, and — critically — **Diebold–Mariano tests** to establish whether accuracy differences are significant rather than noise. Candidate set: seasonal naive benchmark, SARIMA, SARIMAX, ETS, TBATS, Prophet, XGBoost with lag/calendar features, LSTM, and a hybrid (SARIMA on the linear component, ML on the residuals). Deliverable is a fitted, documented forecasting pipeline plus prediction intervals.

**6. Data required**
Monthly export series of adequate length (ideally 180+ observations), plus exogenous monthly series.

**7. Ghanaian / external sources**
UN Comtrade **monthly** trade data (the practical source for monthly frequency); Bank of Ghana external-sector statistics; GEPA NTE annual reports (validation and product definition); GSS CPI/producer price data; World Bank Pink Sheet commodity prices; CHIRPS rainfall.

**8. Unit of analysis**
Commodity × month (national).

**9. Methodology**
Assemble and clean monthly series → test stationarity (ADF, KPSS), seasonality and structural breaks (Bai–Perron) → specify and fit each model class → rolling-origin evaluation with a fixed forecast protocol → Diebold–Mariano comparisons → forecast combination (simple average, inverse-MSE weights) → produce 12-month-ahead forecasts with intervals.

**10. Expected contribution**
*Academic:* the first rigorous, significance-tested forecasting benchmark for Ghanaian horticultural exports; evidence on whether ML actually beats classical methods on short, seasonal African trade series (it often does not — a genuinely publishable negative result). *Practical:* a runnable forecasting tool for GEPA and Bank of Ghana planning.

**11. Feasibility: 7/10**
Method is fully within your toolkit and the evaluation protocol is clean and defensible. The score is held at 7 by one real risk: **monthly** series of sufficient length and quality. Ghana's Comtrade monthly reporting has gaps. Verify series length in week one; if monthly proves unusable, the topic degrades to annual data and becomes too thin to sustain a thesis on its own — at which point fold it in as a chapter of Topic 1 or 5 rather than running it standalone.

**12. Challenges / limitations**
- Short series severely limit deep learning; an LSTM on 150 monthly observations will almost certainly lose, and you should predict that in advance rather than discover it.
- COVID-19 (2020–21) is a violent outlier requiring explicit intervention modelling.
- Forecasting alone can read as thin for a master's thesis; strengthen it by adding the exogenous-driver analysis and a policy-scenario application.

---

### Topic 7 — Post-harvest loss modelling and storage-investment siting

**1. Proposed title**
*Where Ghana Loses Its Harvest: Modelling Post-Harvest Loss Determinants and Optimising Storage Investment Placement*

**2. Research problem / gap**
Post-harvest losses in Ghanaian cereals and perishables are large and routinely cited, but the figures circulating in policy documents are national averages that conceal enormous spatial and seasonal variation. There is no model that predicts *where* and *when* losses concentrate, and consequently no analytical basis for siting storage and aggregation infrastructure. Investment decisions are made on political rather than loss-minimising grounds.

**3. Research questions**
1. What explains variation in post-harvest loss rates across Ghana's regions, crops and years?
2. How much of the variation is attributable to harvest-period weather versus infrastructure and market access?
3. Where are the loss hotspots, once exposure (production volume) is accounted for?
4. Given a fixed budget, where should storage capacity be sited to maximise loss averted?

**4. Variables**
- **Dependent:** post-harvest loss rate (a proportion, bounded 0–1) by region × crop × year; and physical loss volume for the optimisation stage.
- **Independent:** rainfall and relative humidity during and immediately after the harvest window, temperature, production volume, road density and travel time to nearest major market, existing storage capacity, share of production marketed, mechanisation proxy, crop type.

**5. Model to build**
Two components, which is what lifts this above a standard regression thesis.
- *Predictive:* **fractional response regression** (Papke–Wooldridge) or **beta regression** — the correct estimators for a bounded proportion, where OLS would be wrong — plus a random forest benchmark and SHAP interpretation. Spatial dependence tested with Moran's I and, if present, handled with a spatial error/lag specification.
- *Prescriptive:* a **facility-location optimisation** (mixed-integer linear program) that takes the fitted loss surface as input and selects storage sites and capacities to maximise loss averted subject to a budget constraint, solved in Python (`PuLP`/`OR-Tools`). Sensitivity analysis across budget levels produces an investment frontier.

**6. Data required**
Sub-national loss estimates; production volumes; harvest-window weather; road network and travel-time surface; existing storage inventory; unit storage costs.

**7. Ghanaian / external sources**
**APHLIS+** (the key enabler — free, sub-national, modelled loss estimates for cereals and legumes); MoFA SRID (production); CHIRPS/ERA5 (harvest-window weather); National Food Buffer Stock Company and MoFA (existing warehouse inventory); Malaria Atlas Project / OSM-derived travel-time surfaces (accessibility); CSIR-FRI storage-trial literature and Ghana Grains Council for unit costs; GSS for market infrastructure.

**8. Unit of analysis**
Region (or district, where APHLIS resolution allows) × crop × year for the model; candidate storage site for the optimisation.

**9. Methodology**
Extract APHLIS loss estimates → merge production, weather, accessibility → estimate fractional/beta regression and random forest → map predicted loss surface (GIS) → define candidate sites and cost parameters → formulate and solve the MILP → run budget sensitivity → present an investment ranking.

**10. Expected contribution**
*Academic:* a spatially explicit, weather-linked PHL model for Ghana, plus a rarely-attempted linkage from an econometric loss model into a prescriptive optimisation. *Practical:* a defensible, budget-constrained storage-siting recommendation.

**11. Feasibility: 7/10**
The modelling is strong and the two-stage structure is genuinely impressive for a master's thesis. Held at 7 because APHLIS values are themselves *modelled* rather than observed, which weakens the dependent variable, and because credible unit cost data for the MILP is the hardest single input to source.

**12. Challenges / limitations**
- **The central weakness, and you must confront it head-on:** modelling APHLIS outputs risks modelling APHLIS's own model. Mitigate by validating against independent loss measurements from CSIR-FRI trials or published field studies, and by framing the contribution as loss *allocation* and *prescription* rather than loss *measurement*.
- Perishable horticultural crops (tomato, pepper, mango) are poorly covered by APHLIS, which is cereal/legume-focused — this may force a staples focus, narrowing the horticultural angle.
- MILP cost parameters will need assumptions; make them explicit and run wide sensitivity.

---

### Topic 8 — Spatial price transmission and volatility in staple food markets

**1. Proposed title**
*Market Integration, Asymmetric Price Transmission and Volatility Spillover in Ghana's Staple Food Markets*

**2. Research problem / gap**
Ghana's food markets are spatially segmented between northern production zones and southern consumption centres, and there is a persistent policy suspicion that price falls at farmgate transmit to consumers slowly while price rises transmit quickly. Testing that asymmetry formally, and quantifying volatility spillover between markets, has not been done systematically with recent data or with a proper threshold framework.

**3. Research questions**
1. Are Ghana's major staple markets cointegrated, and how strong is spatial integration?
2. How long does a price shock in one market take to transmit to others?
3. Is transmission asymmetric — faster upward than downward?
4. How does volatility spill over between markets and between staples, and did the 2022–23 inflation episode change the structure?

**4. Variables**
- **Dependent:** monthly/weekly wholesale prices for maize, rice (local and imported), cassava, yam, millet, sorghum, groundnut across Accra, Kumasi, Tamale, Techiman, Takoradi.
- **Independent / conditioning:** transport cost proxy (fuel price), exchange rate, seasonality, harvest calendar, world price of imported substitute, policy events.

**5. Model to build**
A layered time-series programme: unit-root testing (ADF, PP, KPSS, and a break-robust test) → Johansen cointegration → **VECM** with impulse response and variance decomposition → **threshold and momentum TAR (TAR/M-TAR, Enders–Siklos)** to test asymmetric adjustment → **multivariate GARCH (BEKK or DCC)** for volatility spillover → optional Markov-switching model for regime changes around the inflation episode.

**6. Data required**
Long monthly price series across multiple markets and commodities.

**7. Ghanaian / external sources**
WFP VAM and FAO GIEWS-FPMA (free, monthly, market-level — the backbone); MoFA/SRID wholesale price bulletins; **Ghana Commodity Exchange** (higher-frequency prices since 2018); GSS CPI food sub-indices; Esoko (commercial, if accessible); Bank of Ghana (exchange rate, fuel).

**8. Unit of analysis**
Market × commodity × month.

**9. Methodology**
Assemble and deflate price panels → stationarity and break testing → cointegration → VECM estimation and diagnostics → asymmetric threshold testing → MGARCH volatility modelling → interpret against transport infrastructure and policy timeline.

**10. Expected contribution**
*Academic:* current-data evidence on Ghanaian market integration with formal asymmetry testing, which most Ghanaian price studies omit. *Practical:* identifies which corridors are poorly integrated, informing where market-information and road investment would pay.

**11. Feasibility: 8/10**
Excellent free data with long series, and the methods are squarely in the Stata/EViews wheelhouse. The reason this is not higher is originality: price transmission is a crowded literature in West Africa, so you would be competing on rigour and recency rather than novelty. The asymmetry and volatility-spillover layers are what keep it defensible.

**12. Challenges / limitations**
- Missing observations in market price series require careful, documented imputation.
- Results are sensitive to lag-length and cointegration-rank selection; report robustness.
- Establishes statistical relationships, not causal mechanisms — resist over-claiming about trader behaviour.
- The topic is only weakly horticultural (staples-dominated), which matters if your programme expects a horticulture focus.

---

### Topic 9 — Competitiveness dynamics against comparator exporters

**1. Proposed title**
*Has Ghana Lost Its Horticultural Edge? Revealed Comparative Advantage Dynamics and Their Determinants Relative to Côte d'Ivoire, Kenya and Costa Rica, 1995–2024*

**2. Research problem / gap**
Ghana's horticultural export performance has been uneven — most conspicuously in pineapple, where the global shift to the MD2 variety in the mid-2000s coincided with a sharp loss of European market share to Costa Rica while Ghana's smallholder base remained on Smooth Cayenne. Ghana's competitiveness is usually discussed anecdotally. What is missing is a formal analysis of whether comparative advantage in specific horticultural lines is persistent or mobile, and what drives movement between competitiveness states.

**3. Research questions**
1. How has Ghana's revealed comparative advantage in major horticultural products evolved relative to comparators?
2. Is comparative advantage persistent or mobile — i.e. what are the transition probabilities between RCA states?
3. What determines RCA levels: exchange rate, yield, infrastructure, certification, FDI, standards compliance?
4. What was the quantitative effect of the mid-2000s varietal transition on Ghana's pineapple export trajectory?

**4. Variables**
- **Dependent:** Balassa RCA and symmetric RCA by product × country × year; RCA state (four classes) for the transition analysis.
- **Independent:** real effective exchange rate, domestic yield, unit export value (quality proxy), port/logistics performance index, FDI in agriculture, certification prevalence, distance-weighted market access, border-rejection intensity (links neatly to Topic 1).

**5. Model to build**
Three linked components.
- **Markov transition probability matrix** over RCA classes, with a test of whether transitions differ across countries and periods, plus the ergodic (long-run) distribution — this converts a descriptive index into an actual stochastic model.
- **Panel regression of RCA determinants** (fixed effects; or a fractional/Tobit specification given RCA's bounded, skewed distribution).
- **Synthetic control** for the pineapple episode: construct a counterfactual "synthetic Ghana" from a donor pool of other pineapple exporters and estimate the export path Ghana would have followed absent the varietal transition, with placebo-based inference.

**6. Data required**
Long bilateral/global trade panel for Ghana and comparators; macro and infrastructure covariates.

**7. Ghanaian / external sources**
UN Comtrade and FAOSTAT (trade); GEPA NTE reports (product definitions and Ghanaian corroboration); World Bank WDI and LPI (macro, logistics); UNCTADstat; CSIR-CRI and academic literature on the MD2 varietal transition; FAGE/SPEG for narrative validation of the pineapple episode.

**8. Unit of analysis**
Country × product × year.

**9. Methodology**
Compute RCA/RSCA series → classify into states and estimate the Markov transition matrix → test for mobility differences → estimate the determinants panel → build the synthetic control for pineapple with placebo tests → integrate into a competitiveness narrative.

**10. Expected contribution**
*Academic:* moves Ghanaian competitiveness analysis beyond index reporting into a stochastic-mobility and quasi-experimental framework; a rare formal estimate of the pineapple varietal shock. *Practical:* evidence on whether Ghana's competitiveness losses are structural (sticky) or recoverable.

**11. Feasibility: 8/10**
Data is entirely Tier A. The synthetic control component is what makes this rigorous rather than descriptive, and it is the part you must not cut. Note that RCA analysis on its own would **not** satisfy a "build a model" requirement — the Markov and synthetic-control layers are what make this a qualifying topic, so treat them as mandatory.

**12. Challenges / limitations**
- Synthetic control with a small donor pool yields weak inference; use placebo tests and report the pre-treatment fit honestly, and be prepared to report an inconclusive result.
- RCA is a coarse, well-known-flawed measure; complement with unit-value and export-survival analysis.
- Dating the varietal transition precisely is difficult and the treatment date is somewhat arbitrary; run sensitivity across plausible dates.

---

### Topic 10 — Cashew value-chain optimisation: raw export versus domestic processing

**1. Proposed title**
*Raw Nuts or Kernels? A Mixed-Integer Optimisation of Domestic Processing Capacity in Ghana's Cashew Value Chain*

**2. Research problem / gap**
Ghana exports a large share of its cashew crop as raw nuts, capturing a fraction of the value that accrues downstream, and policy has repeatedly proposed processing incentives and export restrictions. What has not been done is a quantitative optimisation of *where*, at *what scale*, and under *what price conditions* domestic processing is economically justified — as opposed to asserted.

**3. Research questions**
1. What is the current spatial distribution of cashew production, processing capacity and export flow?
2. Under prevailing prices and costs, what processing capacity and siting maximises net value added subject to supply and capital constraints?
3. How sensitive is the optimal configuration to the raw-nut/kernel price spread, energy cost and labour cost?
4. Under what price conditions, if any, would an export restriction on raw nuts raise national value added?

**4. Variables**
- **Decision variables:** location and capacity of processing plants (binary siting, continuous capacity), raw nut flows from production zone to plant or port, kernel output allocation.
- **Parameters:** raw nut and kernel prices, processing cost per tonne, transport cost, plant capital cost, capacity bounds, production by district, shelling recovery rate, energy and labour costs.
- **Objective:** maximise net value added (or profit) across the chain.

**5. Model to build**
A **mixed-integer linear programme** (facility location plus transhipment), solved in Python with `PuLP` or `OR-Tools`, with binary siting variables and continuous flow variables. Layered on top: **Monte Carlo sensitivity** over uncertain prices and costs, and **scenario analysis** of the raw-nut export-restriction policy. Optionally, a partial-equilibrium or simple price-endogenous extension to avoid the naive assumption that Ghana is a pure price-taker.

**6. Data required**
District-level cashew production; existing processor locations and capacities; raw nut and kernel price series; processing, transport, energy and capital costs; recovery rates.

**7. Ghanaian / external sources**
**Tree Crops Development Authority** (cashew registry, production, processor list); GEPA NTE (export volumes and values); UN Comtrade (raw versus kernel HS lines — HS 0801.31 vs 0801.32 gives you the split directly); CSIR (processing recovery and technical parameters); African Cashew Alliance and Ghana's cashew sector reports (costs); FAGE members for cost validation.

**8. Unit of analysis**
Production district and candidate processing site; commodity flow arc.

**9. Methodology**
Map production and processing → estimate cost parameters from sector reports and, where possible, interviews → formulate the MILP → solve baseline → Monte Carlo over price/cost uncertainty → policy scenarios → present an optimal siting map with an investment schedule.

**10. Expected contribution**
*Academic:* an operations-research treatment of an agricultural value chain, which is uncommon in Ghanaian agribusiness scholarship and a genuine methodological differentiator. *Practical:* a concrete siting and capacity plan, and a defensible answer to a live policy argument about export restrictions.

**11. Feasibility: 6/10**
The model is distinctive and the trade data is free. The score is held down by **cost parameters**: processing cost, capital cost and transport cost per tonne-km are not published anywhere reliable, and without them the optimisation rests on assumptions. This is the topic most exposed to Tier C dependency. Viable if TCDA and sector reports come through; risky otherwise.

**12. Challenges / limitations**
- An optimisation is only as credible as its cost parameters; heavy sensitivity analysis is mandatory and the thesis must be honest that outputs are conditional.
- The MILP assumes a central planner; real investment is decentralised and responds to incentives the model does not contain.
- Price-taking assumption is questionable if Ghana's volumes move the regional market.
- Weakest link to *horticulture* proper if your programme defines that narrowly (cashew is a tree nut).

---

### Topic 11 — Tomato: seasonal glut, import dependence and a decision model

**1. Proposed title**
*Seasonal Glut and Import Dependence in Ghana's Tomato Market: An Integrated Forecasting and Inventory-Optimisation Model*

**2. Research problem / gap**
Ghana simultaneously experiences harvest-season tomato gluts with severe farmgate price collapse and heavy off-season import dependence on Burkina Faso — a well-known paradox that is chronically described and rarely modelled. The gap is an integrated model that forecasts the seasonal supply–price cycle and then determines an optimal storage, processing or staggered-planting response.

**3. Research questions**
1. What is the magnitude and timing of the seasonal tomato price and supply cycle across Ghana's markets?
2. How predictable is the seasonal price trough, and with what lead time?
3. What is the estimated welfare cost of the glut in lost farmgate value?
4. What combination of storage, processing capacity or planting-schedule adjustment minimises that loss, subject to realistic constraints?

**4. Variables**
- **Dependent:** monthly tomato wholesale/farmgate price; import volume from Burkina Faso; estimated marketed surplus.
- **Independent:** production volume and seasonality, rainfall and temperature, cross-border import volume, transport cost, festival/demand seasonality, storage availability.
- **Decision variables (optimisation stage):** storage volume by month, processing throughput, planting-date allocation.

**5. Model to build**
A **seasonal forecasting model** (SARIMAX with harmonic seasonal terms, plus an ML benchmark) for price and supply, feeding a **stochastic inventory / seasonal-allocation optimisation** that chooses storage and processing quantities to maximise expected producer revenue under price uncertainty. Includes a counterfactual welfare calculation of the current glut.

**6. Data required**
Long monthly tomato price series by market; production estimates; cross-border trade volumes; storage and processing cost parameters.

**7. Ghanaian / external sources**
MoFA/SRID and WFP VAM (prices); MoFA SRID (vegetable production, though vegetable coverage is thinner than for staples); UN Comtrade and Ghana Customs (Burkina Faso imports — note that informal cross-border flows are substantially under-recorded); CSIR-FRI (processing and storage parameters); Ghana's tomato processing factories and sector studies (costs).

**8. Unit of analysis**
Market × month; production zone.

**9. Methodology**
Build the seasonal price and supply dataset → decompose seasonality → fit and validate forecasting models → estimate glut welfare loss → formulate and solve the inventory optimisation under price uncertainty → scenario-test interventions.

**10. Expected contribution**
*Academic:* a rare integrated forecast-plus-optimisation treatment of a perishable horticultural commodity in West Africa. *Practical:* directly addresses one of Ghana's most politically salient agricultural failures with a quantified intervention.

**11. Feasibility: 6/10**
High relevance and genuine originality, but the data is materially weaker than for staples: vegetable production statistics are thin, and informal Burkina Faso imports are badly under-recorded, which undermines the supply-side accounting the optimisation depends on. Attractive but risky.

**12. Challenges / limitations**
- Informal cross-border trade is largely invisible in official statistics; this is a serious, not cosmetic, limitation for any supply-balance calculation.
- Tomato production data at monthly frequency essentially does not exist and will have to be inferred, which weakens the whole chain.
- Storage of fresh tomato is technically limited, so the intervention space may be narrower than the model suggests.

---

### Topic 12 — Export survival: how long do Ghana's export relationships last?

**1. Proposed title**
*The Fragility of Ghana's Horticultural Export Relationships: A Duration Analysis of Product–Destination Trade Survival, 1996–2024*

**2. Research problem / gap**
Ghana starts many export relationships and keeps few. Export-promotion policy is overwhelmingly focused on *entry* into new markets, while the evidence internationally is that most new export relationships die within one to two years. Nobody has estimated the survival function of Ghanaian horticultural export spells or identified what makes a relationship durable — which means GEPA's promotion spending has no evidence base on whether entry or retention is the binding constraint.

**3. Research questions**
1. What is the survival profile of Ghana's horticultural product–destination export relationships, and what fraction die within the first two years?
2. Does survival differ by product, destination region and initial export size?
3. What determines hazard of exit: initial value, destination market size, distance, standards stringency, prior experience in the destination, border-rejection history?
4. Does entering a market alongside other Ghanaian exporters of related products improve survival (spillover effects)?

**4. Variables**
- **Dependent:** duration (in years) of a continuous product–destination export spell; the event is spell termination.
- **Independent:** initial export value, product category, destination GDP and distance, common language, tariff, SPS notification intensity, number of other Ghanaian products exported to the same destination, prior spell count for that pair (repeat spells), exchange-rate volatility, global crisis indicators.

**5. Model to build**
**Survival / duration analysis**: Kaplan–Meier survival functions by product and region; Cox proportional hazards with tests of the proportionality assumption; parametric models (Weibull, log-logistic) where proportionality fails; and — because most trade spells are repeated and correlated — a **discrete-time complementary log-log model with random effects** to handle multiple spells per pair and unobserved heterogeneity. Optionally a survival random forest as an ML benchmark.

**6. Data required**
Annual bilateral export flows by product and destination over a long horizon, sufficient to construct spells; standard gravity and standards covariates.

**7. Ghanaian / external sources**
UN Comtrade (the spine — partner-reported imports preferred); GEPA NTE reports (validation of product coverage); CEPII gravity; WITS/TRAINS (tariffs); RASFF (rejection history — links directly to Topic 1); World Bank WDI.

**8. Unit of analysis**
Product × destination × spell.

**9. Methodology**
Construct the rectangular product–destination–year panel → define spells with an explicit rule for gaps (a one-year gap tolerance is standard; test sensitivity) → compute Kaplan–Meier estimates → Cox and parametric hazard models → discrete-time model with frailty → interpret against GEPA's market-development strategy.

**10. Expected contribution**
*Academic:* export-survival analysis is a well-established international literature that has barely been applied to Ghanaian horticulture; duration modelling is also a methodologically distinctive choice that will stand out. *Practical:* a direct, quantified answer to whether GEPA should spend on market *entry* or market *retention* — a genuinely decision-relevant finding.

**11. Feasibility: 8/10**
Data is entirely Tier A with thousands of spells, so power is not a concern. Duration models are well supported in both Stata (`stcox`, `streg`) and R (`survival`). Marked at 8 only because spell construction requires careful, defensible choices that you must document rather than assume.

**12. Challenges / limitations**
- Spell definition is sensitive to the gap rule and to small-value reporting thresholds; run sensitivity analysis.
- Comtrade reporting gaps can create artificial spell breaks; use partner-reported mirror data and check.
- Censoring at both ends of the panel must be handled explicitly.
- Survival analysis identifies correlates of exit, not causes; be careful with policy language.

---

## Part 3 — Ranked top 10

Ranked on the combination you asked for: data availability, model-building potential, academic rigour, practical relevance, originality, and fit to a master's timeframe.

| Rank | Topic | Model class | Data risk | Originality | Feasibility |
|---|---|---|---|---|---|
| **1** | **T1 — EU border-rejection risk model** | Count panel (NB/ZIP) + gradient boosting + SHAP; DiD/synthetic control | **None** (Tier A only) | **High** | **9/10** |
| **2** | **T2 — Climate–yield response & forecasting** | Panel FE with non-linear temperature + ML benchmark + within-season forecast | **Very low** | Medium-high | **9/10** |
| **3** | **T12 — Export relationship survival** | Kaplan–Meier, Cox, discrete-time cloglog with frailty | **None** (Tier A only) | **High** | **8/10** |
| 4 | T5 — Gravity model & export potential | PPML with high-dimensional FE + potential estimation | None | Medium | 8/10 |
| 5 | T3 — Farm technical efficiency | Stochastic metafrontier (Battese–Coelli) + bootstrapped DEA | Moderate (GSS access) | Medium | 8/10 |
| 6 | T4 — Regional DEA & Malmquist TFP | DEA, Malmquist decomposition, Simar–Wilson second stage | Low-moderate | Medium | 8/10 |
| 7 | T9 — RCA dynamics & pineapple counterfactual | Markov transition matrix + panel + synthetic control | None | Medium-high | 8/10 |
| 8 | T8 — Price transmission & volatility | VECM, TAR/M-TAR, MGARCH | Very low | **Low** (crowded) | 8/10 |
| 9 | T7 — Post-harvest loss & storage siting | Fractional/beta regression + ML + MILP optimisation | Moderate | High | 7/10 |
| 10 | T6 — Export earnings forecasting tournament | SARIMAX/ETS/ML/hybrid with Diebold–Mariano | Moderate (monthly series) | Medium | 7/10 |

*Below the line:* **T10 (cashew MILP)** at 6/10 — excellent model, but cost parameters sit in Tier C. **T11 (tomato)** at 6/10 — high relevance and originality, but informal-import invisibility undermines the supply-side accounting the model depends on. Both are strong *if* a specific data relationship materialises; neither should be chosen before that is confirmed.

---

## Part 4 — The top three, in detail

### 🥇 First — Topic 1: Border-rejection risk model

**What you build, concretely.**

*Dataset.* One row per product × year for Ghana plus 5–6 comparator exporters, roughly 2000–2025. Columns: notification count, export volume (exposure), hazard shares, product group, destination composition, harvest-season climate in the sourcing zone, regulatory-event flags.

*Model A — inferential.* Negative binomial panel:

```
E[rejections_pit | X] = exp( β·X_pit + log(exports_pit) + α_p + γ_i + δ_t )
```

with `log(exports)` as an **offset**, so the coefficients are interpretable as effects on the rejection *rate* per unit of trade, not on the raw count. Product, country and year fixed effects. Zero-inflation tested and reported.

*Model B — predictive.* XGBoost classifier taking product, hazard-history, season, exposure, climate and destination features, predicting whether a given product–period generates a rejection, and which hazard class. Trained on pre-2020, tested 2020–2025. Reported as precision-recall AUC with a calibration plot. SHAP values expose the drivers so the model can be defended to a non-technical regulator.

*Model C — causal layer.* Synthetic control on the 2015 vegetable suspension, with Côte d'Ivoire, Nigeria, Kenya, Cameroon and Senegal as donors, plus placebo-in-space tests.

**Inputs:** RASFF notifications, Comtrade volumes, CHIRPS/ERA5 climate, a hand-coded EU regulatory timeline.
**Outputs:** (a) elasticities of rejection rate with respect to volume, climate and regulation; (b) a calibrated risk score for any product–season–destination profile; (c) a ranked risk table usable by PPRSD for inspection targeting; (d) an estimate of whether the 2015 suspension actually improved compliance.

**What the thesis demonstrates.** That you can build a dataset nobody has assembled for Ghana, choose the *correct* estimator for count data with exposure (rather than defaulting to OLS), pair econometric inference with predictive machine learning and explain the difference between them, validate a model properly out of time, and convert model output into an operational decision rule. That combination is well above the median master's thesis.

---

### 🥈 Second — Topic 2: Climate–yield response and forecasting

**What you build, concretely.**

*Dataset.* Region × crop × year panel, ~1992–2024, ≈3,000 rows. Yield from SRID; weather features engineered from CHIRPS and ERA5-Land over crop- and zone-specific growing windows.

*Model A — response.* Two-way fixed effects:

```
yield_rct = f(rainfall) + g(temperature) + Σ dry-spell terms + X'β + α_rc + λ_t + θ_r·t + ε
```

where `g(·)` is a spline or piecewise-linear function of temperature, so a heat-damage threshold is *estimated* rather than assumed. Region-specific trends absorb technology. Standard errors clustered by region, with spatial correction.

*Model B — ML benchmark.* Random forest and XGBoost on identical features, blocked forward-chaining cross-validation, SHAP compared against the panel coefficients. Agreement strengthens both; disagreement is itself reportable.

*Model C — application.* A pre-harvest forecast using only weather observed to a within-season cut-off, scored against climatological and random-walk benchmarks. Then a simple +1 °C / +2 °C projection holding the estimated response fixed.

**Inputs:** SRID regional yields; CHIRPS rainfall; ERA5 temperature; MODIS NDVI; agro-ecological zone boundaries; CSIR agronomic calendars.
**Outputs:** zone-differentiated climate–yield elasticities; an estimated temperature damage threshold; a validated pre-harvest regional yield forecast; a warming-scenario yield projection.

**What the thesis demonstrates.** Serious data engineering (raster-to-panel extraction, boundary harmonisation, agronomic feature construction), correct panel identification, non-linear specification, honest out-of-sample validation, and a policy-usable forecasting product. It is also the *safest* topic here — the climate data will never be withheld from you.

---

### 🥉 Third — Topic 12: Export relationship survival

**What you build, concretely.**

*Dataset.* Every Ghanaian horticultural product × destination pair observed 1996–2024, converted into spells. Thousands of spells, right- and left-censored, with covariates measured at spell start and time-varying.

*Model A — descriptive-inferential.* Kaplan–Meier survival curves by product group and destination region, with log-rank tests. This alone will likely show that a large majority of relationships die within two years — a striking headline finding.

*Model B — hazard.* Cox proportional hazards with Schoenfeld tests; where proportionality fails, Weibull and log-logistic parametric alternatives.

*Model C — the rigorous one.* Discrete-time complementary log-log with **random effects (frailty)**, because the same product–destination pair produces multiple correlated spells and unobserved pair-specific quality would otherwise bias the hazard estimates.

**Inputs:** Comtrade bilateral flows (partner-reported), CEPII gravity covariates, tariffs, SPS notification counts, RASFF rejection history.
**Outputs:** survival functions; hazard ratios for initial export size, destination characteristics, standards stringency and rejection history; a quantified answer on whether entry or retention is Ghana's binding export constraint.

**What the thesis demonstrates.** Command of a model class (duration analysis) that most agribusiness theses never touch, careful handling of censoring and repeated events, and a finding that maps onto a real budget decision at GEPA. It also pairs naturally with Topic 1 — rejection history as a survival covariate — if you want an integrated two-model thesis.

---

## Part 5 — The single strongest recommendation

> ### Topic 1 — *Modelling Sanitary and Phytosanitary Border-Rejection Risk for Ghanaian Agri-Food Exports*

**Why this one, over the other eleven.**

**1. It cannot be blocked.** RASFF and Comtrade are public and downloadable this afternoon. Given that your primary-data access is still uncertain, the deciding criterion should be which topic has *zero* dependence on anyone returning your email. This is that topic. Topic 3 waits on GSS; Topic 10 waits on TCDA; this one waits on nothing.

**2. It satisfies "build and apply a model" twice over.** Not one model but a stack: a count-panel econometric model that yields interpretable elasticities, a machine-learning classifier that yields a calibrated prediction, and a quasi-experimental design that yields a causal estimate. It exercises both halves of your toolkit, and — importantly — the thesis can explicitly discuss *why* you used each, which is exactly the kind of methodological self-awareness that distinguishes a strong thesis from a competent one.

**3. It is genuinely original for Ghana.** The African food-safety rejection literature is largely cross-country and descriptive. An exposure-adjusted, hazard-disaggregated, Ghana-focused risk model with an ML prediction layer does not currently exist. Even the RASFF-to-HS concordance you build is a small standalone contribution others can reuse.

**4. It attaches to a real, documented Ghanaian episode.** The 2015–2017 EU vegetable export suspension gives you a concrete policy event, a natural quasi-experiment, and a narrative that anchors an abstract model in something a Ghanaian reader immediately recognises.

**5. The output is a tool, not a table.** The final artefact is a risk score PPRSD or GEPA could actually use to target inspections. "Practical relevance" is usually the weakest section of a thesis; here it is the strongest.

**6. It positions you well afterwards.** The skill set on display — building a dataset from a regulatory portal, exposure-offset count modelling, temporally validated ML with SHAP, synthetic control — reads equally well to an academic panel and to an employer.

### The one thing to verify before committing

Spend **one day** on this, before anything else: pull Ghana's RASFF notifications and count them by year, product and hazard. You need enough non-zero cells to estimate a panel.

- **If Ghana alone yields several hundred notifications** with reasonable spread across products and years — proceed exactly as specified above.
- **If Ghana alone looks thin** — do not abandon the topic. Widen to a West African panel (Ghana, Nigeria, Côte d'Ivoire, Senegal, Benin, Togo, Burkina Faso) with Ghana as the focal country and country fixed effects. This raises N, makes the comparative claim stronger, and costs you nothing but a slight reframing of the title.
- **If, implausibly, both fail** — switch to **Topic 2**, which is the designated fallback precisely because its data supply is inexhaustible and free.

### Suggested first three weeks

| Week | Action |
|---|---|
| 1 | Scope RASFF counts (the go/no-go check above). **Simultaneously** file the GSS microdata request and the GEPA/FAGE data letters — they cost nothing now and open options later. |
| 2 | Build the RASFF extract and the RASFF→HS concordance. Pull Comtrade exposure data. Assemble the merged panel. |
| 3 | Run descriptive statistics and the first negative binomial specification. Draft the literature review around SPS standards as trade barriers, and lock the proposal. |

Filing the Tier B and Tier C requests in week one is the highest-return thirty minutes in the whole project: if FAGE or PPRSD data arrives in month three, it becomes a powerful validation chapter; if it never arrives, the thesis is already complete without it.

---

## Appendix — Notes and caveats on sourcing

- Every institutional detail above (publication names, agency mandates, dataset coverage) should be **verified against the current source before it goes into your proposal**. Agency structures change — Ghana's regional reorganisation in 2018 and the creation of the Tree Crops Development Authority in 2019 are two examples of changes that invalidate older references.
- Specific figures deliberately are **not** quoted in this document. Any number you cite — export values, loss percentages, prevalence rates — must come from the primary source with a date attached, not from secondary summaries.
- The 2015–2017 EU vegetable suspension, the mid-2000s MD2 pineapple transition, and the region reorganisation are the three factual anchors used most heavily here. Confirm dates and scope from primary documents before building your identification strategy around any of them.

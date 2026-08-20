# FringeNet

**Physics-informed deep learning for thin-film thickness metrology**

반도체 다층 박막 두께 광학 계측 AI — 간섭무늬(fringe)에 인코딩된 두께를 읽는다.

반사율 스펙트럼(226개 파장 채널)으로부터 4층 박막(SiN/SiO₂/SiN/SiO₂ on Si)의 두께를
역산하는 계측 역문제를, 미분가능한 광학 물리 모델(Transfer Matrix Method, TMM)을 결합한
딥러닝으로 푼다. 결과를 한 문장으로:

> **도메인 물리를 모델 설계와 추론에 심으면, 1.52M 파라미터 파이프라인이 140배 큰
> 리더보드 1위 모델(213M)의 재현을 넘는다 — 최종 리더보드 test MAE 0.33895 nm (4위),
> 213M 단독 대비 −29%.**

![헤드라인 — 1.5M CNN + 물리 보정이 213M 단독을 넘는다](reports/figures/fig_headline.png)

- 데이터: [월간 데이콘 — 반도체 박막 두께 분석 경진대회](https://dacon.io/competitions/official/235554/overview/description)
- 키워드: optical metrology, spectral reflectometry, inverse problem, physics-informed ML, differentiable TMM
- 본 문서는 결과 중심의 서사(§1~6)와 방법·재현 상세(§7~)로 나뉜다. 실험별 딥다이브는
  [reports/README.md](reports/README.md)(진행 비교표·리포트 인덱스), 진행 연표는
  [docs/](docs/)가 정본이다 — 본문 수치는 전부 그 정본에서 복사한 실행 산출물이다.

---

## 1. 문제 — 반도체 박막 두께의 비파괴 광학 계측

반도체 공정은 웨이퍼 위에 박막을 쌓고(증착), 깎고(식각), 패턴을 새기는 일의 수백 번
반복이다. 특히 3D NAND는 산화막(SiO₂)과 질화막(SiN)을 수십~수백 층 교대로 적층(ON stack)
하는데, 층 두께의 오차와 불균일이 누적되면 이후 식각 및 셀 특성에 직접적인 수율 문제로
이어진다. 이 프로젝트의 SiN/SiO₂/SiN/SiO₂/Si(기판) 구조는 그 축소 모형이다.

양산 라인에서는 웨이퍼를 자르지 않고 두께를 재야 하므로 **비파괴 광학 계측**을 쓴다.
백색광을 쏘면 각 계면에서 부분 반사된 파들이 광경로차에 따라 파장별로 보강·상쇄 간섭을
일으키고, 반사율 스펙트럼 R(λ)에 두께 정보가 간섭무늬(fringe)로 인코딩된다.

두께 → 스펙트럼의 **순방향 계산은 TMM으로 정확하고 빠르다.** 병목은 역방향이다 —
전통적 라이브러리 피팅은 층이 많아질수록 해의 다중성·국소최소·속도 문제를 겪는다.
데이터는 (두께 4값, 스펙트럼 226채널) 쌍 81만 행(train)과 스펙트럼만 있는 1만 행(test),
평가지표는 층별 두께 MAE[nm]다.

## 2. 비교 기준 — 리더보드 1위 모델(213M)을 직접 재현해 상한으로 삼았다

남의 리더보드 점수가 아니라 **직접 재현해 같은 평가셋에서 잰 수치**와 싸웠다. 대회 1위
단일 모델(213.2M 파라미터 skip-MLP)을 원본 그대로 재현하면 holdout MAE
**0.3955 nm**다 ([reports/strong_baseline.md](reports/strong_baseline.md)). 이것이 이
프로젝트의 상한 기준선이고, 질문은 하나다 — **도메인 물리를 아는 작은 모델이 이 순수
용량의 벽에 어디까지 가는가.** 최종 채택 백본(1.52M)과의 크기 차이는 140배다.

## 3. 최종 방법에 도달한 네 걸음

각 걸음은 「관찰 → 설계 결정 → 측정된 효과」의 반복이다. 전체 사다리는 위 헤드라인 그림,
수치 정본은 [reports/README.md](reports/README.md)의 진행 비교표다.

### 걸음 1 — 스펙트럼은 순서 있는 신호다: 1D CNN inductive bias (−49%)

파장축이 비식별화되어 실제 nm 값을 모르더라도, 채널 순서가 연속 스펙트럼이라는 구조는
살아 있다. 226채널을 순서 없는 피처로 보는 MLP(4.599 nm) 대신 1D CNN을 쓰되, ablation이
가려낸 조건이 물리와 정확히 대응한다: **수용영역이 전 대역을 덮어야 하고**(dilated,
RF 259 — fringe 주기는 두께에 따라 크게 변한다) **파장축 위치가 보존되어야 하며**(flatten —
같은 무늬라도 어느 파장에 있는지가 두께 정보다) 출력은 물리 범위 [10, 300] nm로
**bound**한다. 이 조합이 2.346 nm(−49%)이고, 조건을 하나라도 어긴 소박한 conv+GAP는
오히려 MLP보다 나쁘다 — 구조 bias는 물리에 맞을 때만 작동한다
([reports/level1_cnn.md](reports/level1_cnn.md)).

### 걸음 2 — 물리 디코더를 데이터에 맞춘다: TMM 캘리브레이션 (자유 파라미터 7개)

파장축과 박막 물성이 미지이므로, 물리를 쓰려면 먼저 forward 모델의 미지수를 데이터에서
식별해야 한다(시스템 식별). 핵심 설계 원칙은 **물리 법칙을 자유도 개수로 강제한다** —
물성 n(λ)는 매끈한 분산 법칙(Sellmeier)을 따라야 하므로 채널별 자유 곡선을 절대 두지
않고, 문헌 광학상수(SiO₂ Malitson · SiN Luke · Si Schinke)를 뼈대로 전체 자유 파라미터를
**7개**로 제한했다. 부수 성과로 데이터의 노이즈를 완전히 규명했다: σ = 0.008658의 가산
노이즈이고 **유계**(|ε| ≤ 0.0152)라서, 이 상한을 넘는 재구성 잔차 하나하나가 통계 없이
모델 오류의 증거가 된다 — 이후 모든 판정의 자(尺)다. 결과: 재구성 RMSE 1.106σ, 디코더
역해 정확도 상한 0.340 nm. 사전 선언한 게이트 6종 중 **(b) 유계 위반율은 통과하지 못했고
(9.99%), 그 사실을 지우지 않고 리스크를 수용한 근거와 함께 기록했다** (§7.2,
[reports/stage_a.md](reports/stage_a.md)).

### 걸음 3 — 물리의 올바른 자리: 손실은 기각, 추론 후 보정은 −74%

physics-informed ML의 통념대로 물리를 **학습 손실**(cycle-consistency 항)로 걸었다 —
그리고 사전등록한 ablation 세 축(무작위 split · held-out 두께 split · warm start) 전부에서
**기각됐다**: 손실 계수 β에 단조로 해롭고, 적합 수준을 맞추면 이득이 0이다
([reports/stage_b.md](reports/stage_b.md)). 같은 동결 디코더를 **추론 후 보정**으로 옮기면
얘기가 다르다: CNN 예측을 출발점으로 관측 스펙트럼에 Levenberg–Marquardt 역산을 돌리면
2.346 → 0.611 nm(−74%)이고, 격자 중앙에서 출발한 같은 LM은 77.7 nm로 실패한다. 즉 역할이
갈린다 — **신경망은 해의 다중성(올바른 fringe 분지)을 풀고, 물리는 분지 안의 정밀화를
맡는다.** 여기에 라벨 없는 **되돌림 규칙**(재구성 잔차가 문턱을 넘는 행은 보정을 버리고
CNN 예측 사용 — LM은 잘못된 분지 바닥까지 성실하게 내려가므로)을 얹은 파이프라인이
0.66M 백본 기준 0.3880 nm로 **213M 단독(0.3955)을 처음 넘었다**
([reports/inversion_refine.md](reports/inversion_refine.md) ·
[reports/cnn_recipe.md](reports/cnn_recipe.md)).

### 걸음 4 — 남은 격차는 순수 적합력: 잔차 연결 + 깊이 ×2 + rFFT 분기 (1.52M)

일반화 격차가 213M과 같게 측정됐으므로 남은 병목은 정규화가 아니라 **용량(적합력)**이다
(train 손실 1.475가 순차 스택의 벽). 구조 → 용량 → 부착 모듈 순서로 최적화했다:
잔차 연결이 같은 파라미터 수에서 벽을 뚫고, 깊이 ×2(블록 10)가 폭 ×2를 3배 차로 이기며
(작은 모델이 버는 방향은 깊이다), 간섭무늬가 주기 신호라는 물리 근거로 rFFT 입력 분기를
붙였다. 최종 백본 `d2-fft`(1.52M)는 단독 holdout 0.3589, 물리 보정 후 **0.3396 nm**다
([reports/task8.md](reports/task8.md)).

## 4. 최종 비교 — 213M 단독 vs 작은 모델 단독 vs 작은 모델 + 물리

세 경쟁자를 같은 holdout(81,000행)과 리더보드 test(1만 행, **격자 밖 연속 두께** — §5)
에서 비교한다. 수치 정본: [reports/task8_judge.md](reports/task8_judge.md) ·
[reports/leaderboard.json](reports/leaderboard.json) ·
[reports/task8_bench.md](reports/task8_bench.md).

| | 파라미터 | holdout MAE [nm] | test MAE [nm] (격자 밖) | 전이 |
|---|---|---|---|---|
| 213M skip-MLP 단독 (1위 재현) | 213.2M | 0.3955 | 0.4752 | +20% |
| d2-fft 단독 (raw CNN) | 1.52M | 0.3589 | 0.5911 | +65% |
| **d2-fft + LM 역산 + 되돌림 (채택)** | **1.52M + 물리 7** | **0.3396** | **0.33895 (4위)** | **−0.2%** |

자원 비용까지 함께 보면 ([reports/task8.md](reports/task8.md) 자원 미터,
정본 [task8_bench.md](reports/task8_bench.md) — Colab 한 세션 CPU·L4 실측):

| 미터 | 채택 파이프라인 | 213M 단독 | 배수 |
|---|---|---|---|
| 파라미터 | 1.52M | 213.2M | **140배 작다** |
| 체크포인트 | 6.1 MB | 813.6 MB | 133배 작다 |
| 학습 (100에폭, L4) | 약 54분 | 199분 | 3.7배 빠르다 |
| 추론 (CPU, ms/행) | 0.999 | 0.705 | 1.42배 느리다 |
| 추론 (L4, ms/행) | 0.070 | 0.047 | 1.49배 느리다 |
| **정확도 (holdout)** | **0.3396** | 0.3955 | **−14%** |
| **정확도 (test, 격자 밖)** | **0.33895** | 0.4752 | **−29%** |

요약하면: **학습 비용·모델 크기는 두 자릿수 배로 작고, 추론은 물리 역산(LM)을 포함해도
동급 수준이며**(CNN forward 단독은 213M보다 빠르다 — L4 0.39배; 합계가 느린 것은 LM
몫이고 배수는 기계와 함께만 인용한다), **정확도는 holdout에서 −14%, 실전 조건인 격자 밖
test에서 −29% 우세하다.** 추론의 LM 비용도 해석적 야코비안 + 조기 종료로 중앙차분 대비
10~16배를 걷어낸 결과다 ([reports/inversion_bench.md](reports/inversion_bench.md)).

## 5. 핵심 발견 — 격자 밖 반전: 물리의 값어치는 정밀도가 아니라 강건성이다

train의 두께는 10 nm 격자 위 전수 조합(30⁴)이고 **test는 격자 밖 연속 두께**다 — 실제
웨이퍼의 두께가 그렇듯이. 이 분포 이동이 최종 판정을 뒤집었다.

![격자 밖 반전 — raw는 전부 열화, 물리 파이프라인만 전이](reports/figures/fig_offgrid.png)

holdout 최강은 SE 어텐션을 단 raw CNN(0.2954)이었지만 test에서 0.5461로 **+85% 붕괴**했다.
측정한 raw 신경망 셋 전부 — 우리 두 모델과 **213M 1위 재현까지** — 격자 밖에서 열화했고
(+20~85%), **물리 파이프라인만 열화 없이 전이됐다**(−0.2%). 정확도가 좋아질수록 raw
신경망은 학습 격자의 구조를 흡수하고, 그 이점은 연속 두께에서 사라진다 — holdout의 raw
우위는 계측 성능이 아니라 격자 보간 인공물이었다.

이 붕괴는 제출 전에 **라벨 없이** 예측 가능했다: 예측을 물리 디코더로 되비춘 재구성
잔차의 holdout↔test 꼬리 비교가 격자 과적합을 검출한다(신호와 열화가 단조 — 꼬리 1.95배 →
+65%, 2.3배 → +85%). 같은 잔차가 행 단위 신뢰도 지표(계측 이상 감지 관점, §7.6)와 되돌림
규칙의 지목 신호로도 쓰인다. 단 이 관문의 검출 범위는 격자 끌림 붕괴까지다 — 213M은
관문을 통과하고도 +20% 열화했으므로(잔차는 노이즈 바닥 아래 sub-nm 오차에 둔감),
**raw 제출 자체를 하지 않는 것**이 규칙이다.

물리 결합의 값어치가 프로젝트를 관통하며 두 번 재정의됐다: 손실이 아니라 **추론**에
있고(걸음 3), 추론에서도 정밀도 회복이 아니라 **분포 이동에 대한 강건성 + 라벨 없는
신뢰도**에 있다(이 절). 채택이 holdout 최강 CNN이 아니라 **파이프라인 최강**인 이유다.

## 6. 한계

- **물리 디코더의 계통오차가 남은 바닥이다.** 게이트 (b) 미통과(§7.2)가 가리키는 물성
  모델 형태의 부족이 post-LM 정확도 바닥(≈0.334 nm)을 만든다 — 물성 모델 개선이 남은
  유일한 레버이고, 마감 내 범위 밖으로 기록했다.
- **등가 분지 실패는 잔존한다** (채택 모델 0.07%). 되돌림 규칙이 줄이는 것은 실패
  건수가 아니라 심각도이고, 남는 실패의 상당수는 관측을 거의 같게 설명하는 등가 분지라
  재시도로 잡히지 않는다.
- **모델링 가정**: 수직입사·등방성·평행 평면층·거칠기 없음. 실제 계측은 유한 개구각과
  거칠기 보정을 포함한다.
- **되돌림 문턱은 transductive다** — 행 집합의 잔차 분포에서 만들므로 단행 배포에는
  미리 계산한 문턱이 필요하다.
- 평가가 시뮬레이션 격자 데이터의 함정(스냅·클리핑·보간 누설)에 오염되지 않도록 통제한
  규약은 §7.5 — 격자 스냅과 범위 클리핑의 holdout 이득은 인공물이라 채택하지 않았다.

---

*이하는 방법·재현 상세다 — 기술 독자용.*

## 7. 방법 상세

### 7.1 미분가능 TMM forward 모델과 물리 단위 테스트

수직입사에서 층 j의 위상두께와 특성행렬(Abelès, Macleod 관례):

$$\delta_j=\frac{2\pi n_j d_j}{\lambda},\qquad
M_j=\begin{pmatrix}\cos\delta_j & i\sin\delta_j/n_j\\ i\,n_j\sin\delta_j & \cos\delta_j\end{pmatrix}$$

$$\binom{B}{C}=\Big(\prod_{j=1}^{4}M_j\Big)\binom{1}{n_s},\qquad
r=\frac{n_0B-C}{n_0B+C},\qquad R=|r|^2$$

구현(`src/physics/tmm.py`)은 배치·파장 벡터화, complex, 미분가능이고 다음 8종 테스트가
전부 통과해야 다음 단계로 진행했다 (`tests/test_tmm.py`):

| 테스트 | 기대 결과 |
|---|---|
| 무층 극한 (d=0) | r = (n₀−n_s)/(n₀+n_s) — 맨 기판 프레넬 반사 회복 |
| 에너지 보존 (무흡수 스택) | R + T = 1, T = 4n₀Re(n_s)/\|n₀B+C\|² |
| λ/4 무반사 | n₁=√(n₀n_s), d=λ/4n₁ → R ≈ 0 |
| Airy 대조 (단층) | 해석해 r=(r₀₁+r₁₂e^{−2iδ})/(1+r₀₁r₁₂e^{−2iδ})와 일치 |
| 미분가능성 | dR/dd가 유한차분과 일치 (autograd 검증) |
| 흡수 기판 | 복소 n_s에서 R<1, NaN/Inf 없음 |
| 층 순서 고정 (비대칭 2층) | 재귀 프레넬 해석해와 일치 — 적층 순서 반전을 검출 (명세 6종이 순서를 고정하지 못해 보강한 7번째) |
| 해석적 야코비안 dR/dd | **autograd와** 일치 (rtol 1e-11, 유한차분은 절단오차가 있어 참조값이 못 된다) + 반환 R은 forward와 비트 동일 + L=1 별도 — 추론 시 LM 역산(§3 걸음 3)의 근거 |

### 7.2 Stage A — 캘리브레이션 (시스템 식별)과 판정 게이트

학습셋의 (두께, 스펙트럼) 정답 쌍으로 forward 모델의 미지수를 역으로 피팅한다
(`src/calibrate.py`, 2단계: 두께축 주파수 식별로 λ 초기값 → 신뢰영역 최소제곱).

| 대상 | 모델 | 출처 | 자유 |
|---|---|---|---|
| λ(c) | 1/λ = ν₀(1 + r₁u + r₂u²), u = c/225 | 두께축 주파수 식별 (닫힌형) | 0 또는 3 |
| SiO₂ | Sellmeier 3항 **동결** | Malitson 1965 | **0 (게이지)** |
| SiN | Sellmeier 2항 | Luke et al. 2015 | 1~2 (B₁, C₁) |
| Si 기판 | 실측표 + 에너지축 3차 스플라인 | **Schinke 2015** (대안: Aspnes 1983 / Green 2008) | 0~2 (ΔE, k 스케일) |

- **게이지 축퇴 — n과 λ는 동시에 식별되지 않는다.** δ = 2πnd/λ 이므로 n과 λ를 같은
  배수로 스케일해도 스펙트럼이 불변이다. **SiO₂를 문헌값에 고정**하는 것이 그 선언이고,
  λ의 절대 스케일은 이 가정에 의존한다 (Si 임계점을 앵커로 쓴 검정은 통과).
- 데이터가 TMM으로 생성되었다는 보장이 없으므로 게이트로 판정했다. **재구성 잔차만 보는
  게이트는 계통오차를 파라미터로 흡수하는 유연한 모델을 항상 유리하게 만들므로**
  파라미터의 물리성과 예측력을 함께 본다 (`scripts/diagnose_calibration.py`):

| | 기준 | 성격 |
|---|---|---|
| (a) 재구성 RMSE | < 1.2σ = 1.039×10⁻² — 노이즈가 있어 완벽한 모델도 σ 아래로 못 내려간다. 1.2라는 배수에 유도는 없으므로 **계통오차 √(RMSE²−σ²)를 함께** 1차 지표로 읽는다 | pass/fail |
| (b) **유계 노이즈 위반율** | 노이즈가 \|ε\| ≤ 0.0152로 **유계**임이 확정됐으므로, 잔차가 이를 넘는 관측은 통계 없이 모델 오류의 증거. 완벽한 모델은 0% | pass/fail |
| (c) 잔차 백색성 | 두께·채널에 구조 없음 — 단독으로는 판별력이 없다 | 참고 |
| (d) 두께 nm 역해 MAE | 디코더를 LM으로 역해한 층별 오차 — 물리가 강제할 수 있는 정확도의 상한 | 실용 판단 |
| (e) 채널 홀드아웃 | 피팅에서 뺀 20채널의 R을 예측 — 매끈한 물리 분산만 통과 가능 | 물리성 증명 |
| (f) 파라미터 물리성 | 문헌 범위 내 + 곡선 매끈 + 독립 두 방법(주파수 식별 vs TMM 최소제곱) 일치 | 물리성 증명 |

> **판정 결과 — (b)는 통과하지 못했고, 그럼에도 TMM을 채택했다.** 사전에 pass/fail로
> 선언했으므로 이 사실을 여기에 적어 둔다: 최선 모델의 위반율이 **9.99%**(완벽한 모델은
> 0%)이므로 **forward 모델은 불완전하다**. fallback(NN emulator)으로 가지 않은 근거는
> 둘이다 — ① 잔여 오차가 용도에는 충분히 작다(역해 MAE **0.340 nm** vs 당시 규제 대상
> CNN 2.346 nm로 7배 여유) ② 지배 성분이 모형이 아니라 **문헌표 불일치**로 측정됐다
> (세 Si 표의 차이가 유계 예산의 70%를 쓴다). NN emulator로 갈아타면 이 잔차는 사라지지만
> (e)·(f)를 함께 잃으므로 목표에 역행한다. **수용하는 리스크**: 물리 항이 0.34 nm 수준의
> 계통 편향을 가진 기준으로 예측을 당긴다 — Stage B ablation이 이 리스크를 측정했고 실측
> 손해의 방향과 일치한다 ([reports/stage_b.md](reports/stage_b.md) §4).

결과와 한계는 [reports/stage_a.md](reports/stage_a.md).

### 7.3 Stage B — 물리 손실 (사전등록 ablation — 기각)

캘리브레이션된 디코더를 **동결**하고 cycle-consistency 항을 학습 손실로 걸었다:

$$\mathcal{L}=\mathrm{MAE}(\hat d, d)+\beta\,\lVert R_{\mathrm{TMM}}(\hat d)-R_{\mathrm{obs}}\rVert_1$$

β ∈ {0, 30, 100, 300}을 세 축(무작위 split · held-out 두께 값 split · 수렴 후 warm start)
에서 대조한 결과 **전 축에서 기각됐다** — val MAE가 β에 단조로 나빠지고, 적합 수준을
맞추면 남는 이득이 0이다. 물리 항은 자기 목적함수(재구성)는 개선하므로 실패는 **정렬**의
문제다 — 관측이 R(d) + ε라 손실 항이 지도 항의 노이즈 낀 대리이고, 게이트 (b)를 못 넘은
디코더의 계통 편향으로 예측을 당긴다. 사전등록한 U자 예측이 어긋난 기록까지
[reports/stage_b.md](reports/stage_b.md)에 있다.

### 7.4 추론 시 물리 — LM 역산 refinement + 라벨 없는 되돌림 규칙 (채택)

같은 동결 디코더를 추론 후 보정으로 쓴다 (`src/physics/invert.py` ·
`scripts/refine_inversion.py`): CNN 예측 $\hat d$를 출발점으로 관측 R에 배치
Levenberg–Marquardt로 재적합한다 — 라벨도 격자도 쓰지 않으므로 test·실계측에 그대로
적용된다. 되돌림 규칙(`flag_unreliable`)은 재구성 잔차가 중앙값 + 5·robust σ를 넘는 행에서
보정을 버리고 CNN 예측을 쓴다 — 지목·문턱 모두 라벨 없이 만들어지고, k = 5는 사전등록
값이다(MAE를 보며 고르면 평가셋 선택이 된다). 비용은 해석적 야코비안(물리 테스트 8번이
근거) + 조기 종료로 중앙차분 대비 10~16배 절감 — 정확도 손실 0
([reports/inversion_refine.md](reports/inversion_refine.md) ·
[reports/inversion_bench.md](reports/inversion_bench.md)).

### 7.5 평가 프로토콜 (정직성 규약)

학습 데이터가 두께 격자를 전수 조합한 **시뮬레이션 산출물**이라는 점은 평가를 왜곡할 수
있다. 포트폴리오로서 신뢰를 얻으려면 이 함정을 스스로 드러내고 통제하는 편이 낫다.

1. **격자 스냅은 분리 보고 — 단 train/holdout에 한해서다.** train 정답이 10 nm 격자
   위에 있으므로 holdout 예측을 최근접 격자로 반올림하면 MAE가 인위적으로 떨어진다
   (실측: 2.346 → 1.287). 이는 계측 성능이 아니라 **데이터 생성 방식의 누설**이므로
   raw MAE와 분리해 보고한다. **test는 격자 밖이라 제출에 스냅을 쓰면 MAE가 약 +1.2 nm
   나빠진다** — 같은 후처리가 holdout에서는 누설이고 test에서는 순손실이다. 같은 계열인
   범위 클리핑(holdout −0.0076 이득)도 격자 끝 쏠림의 인공물이라 채택하지 않았다.
2. **격자 밖 일반화 — 이미 test가 그 평가셋이다.** holdout(격자 위 조합 보간) vs
   test(격자 밖 연속)의 대비로 직접 측정한다 — §5가 그 실측이고, 제출 전 라벨 없는
   관문은 `scripts/check_submission_transfer.py`다.
3. **held-out 두께 값 split** — 무작위 split은 모든 두께 값이 학습에 등장하므로 "조합
   보간"만 측정한다. 특정 두께 값을 통째로 빼는 split을 추가해 외삽 능력을 본다
   (`data.holdout_thickness` — Stage B 라운드 2·3이 이 split이다).
4. **노이즈 강건성** — 입력에 노이즈를 주입한 조건의 열화 곡선을 함께 보고한다. 주입은
   데이터와 같은 종류인 균등 ±0.015가 기본이고(σ = 0.008658 위에 더하는 추가분임을 명시),
   실측은 [reports/cnn_recipe_axes.md](reports/cnn_recipe_axes.md).

### 7.6 부산물 — 계측 신뢰도 지표

추론 시 TMM 재구성 잔차가 큰 샘플은 모델이 확신하지 못하는 측정으로 플래깅할 수 있다 —
실제 fab의 계측 이상 감지(FDC) 관점과 맞닿아 있고, **라벨을 쓰지 않으므로** test·실계측에
그대로 적용된다. 실측(순위상관 ρ 0.70 · 잔차 상위 10%의 실제 오차 4.97 vs 하위 0.60 nm)은
[reports/cnn_recipe_axes.md](reports/cnn_recipe_axes.md). 같은 잔차가 §7.4 되돌림 규칙의
지목 신호이자 §5 제출 전 관문이다.

## 8. 데이터

| 파일 | 내용 |
|---|---|
| `train.csv` | `layer_1`~`layer_4` = 각 층 두께[nm] (타깃) + `0`~`225` = 파장 채널별 반사율 |
| `test.csv` | `0`~`225` 반사율만 제공 |
| `sample_submission.csv` | 제출 양식 (`layer_1`~`layer_4` 예측값) |

- **소자 구조**: 공기 / SiN(layer_1) / SiO₂(layer_2) / SiN(layer_3) / SiO₂(layer_4) / Si 기판
- **평가지표**: 층별 두께의 MAE
- **파장축 비식별화 주의**: 헤더 `0`~`225`는 비식별화된 인덱스로, 실제 파장[nm]이 아니다.
  다만 채널 순서는 연속 스펙트럼으로서 물리적 의미를 가진다(1D conv의 축으로 쓸 수 있다).
- **다운로드**: 대회 페이지에서 규정 동의 후 내려받아 `data/raw/`에 배치한다.
  대회 규정상 데이터는 이 저장소에 포함하지 않는다(`.gitignore` 처리).

### 8.1 검증으로 확정된 사실

`scripts/verify_data.py`(계약 검증)와 `scripts/eda.py`(EDA)의 실제 산출에 근거한다.
유도 과정과 전체 분석은 [reports/eda_notes.md](reports/eda_notes.md),
발견 당시의 기록은 [docs/week_1.md](docs/week_1.md) 참조.

| 사실 | 내용 |
|---|---|
| 두께 격자 | **train만** 각 층 {10, 20, …, 300} nm — 30값 × 4층 = **30⁴ = 810,000행 전수 조합** (중복·결측 없음) |
| **test는 격자 밖** | test 10,000행의 두께는 **격자 위에 있지 않다 — 연속이다.** 라벨이 없어 R에서 역추정했고 독립 두 방법이 일치한다: ① 유계 노이즈 반증(같은 두께라면 `max\|ΔR\| ≤ 2a = 0.0304`인데 train 전수 중 최소가 0.063~0.103), ② 디코더 역해 격자거리 평균 **2.42 nm**(균등 이론 2.5) vs train 0.37 nm |
| 노이즈 | 반사율에 **σ = 0.008658의 가산 노이즈** (균등분포에 가까움, 채널에 균일, **유계** \|ε\| ≤ 0.0152). 음의 반사율이 값의 0.35%, 행의 46.9%에 등장 — 물리적으로 불가능하므로 노이즈의 증거 |
| 층별 가시성 | 10 nm 변화의 SNR 최소 10.3 — **원리적 사각지대 없음.** 층별 오차 격차는 모델 문제로 해석 |
| 채널별 정보량 | 대역 오른쪽 끝이 왼쪽의 약 3배 (노이즈는 균일하므로 SNR도 3배) |
| fringe | 두께↑ → 무늬 조밀 (정성 확인까지만 — 비식별 파장축에서 정량 법칙은 주장하지 않는다) |

## 9. 저장소 구조

```
.
├── README.md                   # 본 문서 — 결과 서사(§1~6) + 방법·재현 상세(§7~)
├── CLAUDE.md                   # Claude Code 작업 메모리 (계약·테스트 스펙·백로그)
├── requirements.txt
├── docs/                       # 주차별 실험 노트 — 진행·결과·발견·결정·TODO (§11)
├── configs/                    # 실험 설정 — runs/와 같은 2단 구조 (§11)
│   └── <실험>/<변형>.yaml      #   예: mlp_baseline/dropout0.0.yaml
├── notebooks/                  # Colab GPU 드라이버 — 라운드별 1개, 완료 후 실행 로그 불변
│   └── <대실험>/roundN_<내용>.ipynb
├── data/                       # 대회 데이터 — 파일은 git 미포함, 구조만 .gitkeep (§8)
│   ├── raw/                    #   데이콘 원본 (사용자가 직접 배치)
│   └── cache/                  #   parquet 캐시 (최초 실행 시 자동 생성)
├── runs/                       # 실행 산출물 — **텍스트 2종만 git 추적** (§11)
│   ├── CHECKPOINTS.md          #   Drive 미러 목록·sha256·복구 방법
│   └── <실험>/<변형>/          #   train.log · metrics.json (+ stage_a만 model.pt — §11)
├── reports/
│   ├── README.md               # **인덱스** — 진행 비교표·읽기 순서·정본/취합 구분
│   ├── <실험>.md               # 대실험별 취합 리포트 — 판단·서사
│   ├── *_gate.md · *_judge.md · *_axes.md · *_bench.md · *_metrics.md · leaderboard.json
│   │                           # 산출 정본 (재실행 시 덮어씀 — leaderboard.json만 손기록)
│   └── figures/                # 산출 그림 (리포트·인덱스가 임베드)
├── scripts/                    # 전부 산출물 생성기 — 리포트 수치의 재현 경로다
│   ├── verify_data.py          # 데이터 계약 검증 (+ --deep: test 격자 밖 반증)
│   ├── eda.py                  # EDA 그림 3종 + 측정값 (reports/eda_metrics.md)
│   ├── measure_noise.py        # 노이즈 σ·유계 상한 측정 (채널축 m차 차분)
│   ├── diagnose_calibration.py # Stage A 게이트 (a)~(f) 진단 + 그림 (§7.2)
│   ├── diagnose_predictions.py # 예측 오차 구조 진단
│   ├── evaluate_axes.py        # 평가 축 — 노이즈 강건성(§7.5) + 신뢰도 지표(§7.6)
│   ├── refine_inversion.py     # 역산 refinement 판정 (§7.4)
│   ├── judge_recipe.py         # post-LM·분지 실패율·되돌림 판정 (*_judge.md)
│   ├── bench_invert.py         # 역해 LM 추론 비용 벤치 (*_bench.md)
│   ├── check_submission_transfer.py  # 제출 전 관문 — 잔차 holdout↔test 전이 (§5)
│   ├── analyze_stage_b_curves.py  # 적합 수준 맞춘 β 대조 (stage_b_curves*.md)
│   ├── make_headline_figure.py # 헤드라인·격자 밖 그림 — 수치는 산출물에서 읽는다
│   └── check_notebook_regression.py  # 노트북 옛 버퍼 되돌림 탐지 (커밋 전)
├── src/
│   ├── physics/
│   │   ├── tmm.py              #   미분가능 TMM + 해석적 야코비안 — 프로젝트의 물리 코어
│   │   ├── invert.py           #   배치 LM 역산 + 되돌림 규칙 (§7.4의 코어)
│   │   ├── dispersion.py       #   문헌 광학상수 로더·Sellmeier·에너지축 스플라인
│   │   ├── freq_id.py          #   두께축 주파수 식별 — λ축의 닫힌형 복원
│   │   └── literature/         #   refractiveindex.info 원본 파일 (CC0, git 추적)
│   ├── data/dataset.py         # CSV → parquet 캐시 → numpy/torch
│   ├── models/                 # 모델 레지스트리·팩토리 (__init__.py의 build_model)
│   │   ├── mlp.py              #   baseline MLP — 구조 bias 없는 대조군
│   │   ├── heads.py            #   공용 출력단 (ThicknessBound 등)
│   │   ├── cnn.py              #   1D CNN — flatten·dilated·bound·residual·SE·rFFT 플래그 (채택 백본)
│   │   ├── convnext.py         #   ConvNeXt-1D (Task 8 대조군)
│   │   └── winner_skip_mlp.py  #   1등 솔루션 213M skip-MLP 충실 재현 (상한 기준선)
│   ├── utils/                  # 시드 고정·원자적 저장
│   ├── calibrate.py            # Stage A 캘리브레이션 — 물리 제약 최소제곱 TRF (자유도 1~7)
│   ├── losses.py               # Stage B 물리 손실 — 동결 TMM 디코더 + beta 워밍업 (§7.3)
│   ├── train.py                # baseline 학습 — CPU 경로
│   ├── train_gpu.py            # GPU(Colab) 학습 경로 — resume+Drive 미러
│   └── evaluate.py             # holdout 재평가·제출 파일 생성 (--refine = 물리 보정 경로)
└── tests/                      # 물리 단위 테스트(§7.1)·역산·로더·모델·학습·노트북 규약
```

## 10. 시작하기

```bash
# Python >= 3.11 (pyproject.toml). venv 또는 conda 환경 어느 쪽이든 된다.
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest -q                          # 물리 단위 테스트 + 로더·역산·노트북 규약 테스트
python scripts/verify_data.py      # 데이터 계약 검증 (통과 시 종료 코드 0; --deep = 격자 밖 반증)
python scripts/eda.py              # EDA 그림 3종 + 측정값 표
python -m src.train --config configs/mlp_baseline/dropout0.0.yaml  # baseline 학습 (CPU)

# GPU가 필요한 학습(CNN 이후)은 Colab에서 노트북 Run-All로 돌린다:
#   notebooks/<대실험>/roundN_<내용>.ipynb
# 로컬 CPU 스모크:
python -m src.train_gpu --config configs/level1_cnn/flatten-dilated-bound.yaml \
  --device cpu --subset 20000 --epochs 2 --run-name smoke --no-resume

# 최종 제출 경로 — 채택 모델 + 물리 보정(LM 역산 + 되돌림 규칙)까지 한 줄.
# holdout 확정 수치와 test 보정 제출 csv가 함께 나온다 (체크포인트는 runs/CHECKPOINTS.md 참조).
python -m src.evaluate --run runs/task8/d2-fft --submission --refine
```

`verify_data.py` 최초 실행은 `train.csv`(1.9 GB)를 파싱해 `data/cache/train.parquet`을
만드느라 약 30초 걸리고, 이후 실행은 캐시를 읽어 3~4초다.

**메모리 요구(실측 최대 상주)**: `verify_data.py`·`eda.py` 약 4 GB, `src.calibrate`·
`diagnose_calibration.py` 약 5 GB — 두께축 주파수 식별의 조건부 평균이 holdout 제외
train 전체(73만 행)를 올리기 때문이다. 8 GB 미만 환경에서는 Stage A 재현이 스왑을 탄다.

## 11. 문서·실험 관리

본문(§1~7)의 수치는 전부 [reports/](reports/README.md)의 산출 정본에서 복사한 것이고,
수치가 갱신되면 진행 비교표·헤드라인 그림과 함께 갱신한다. 역할 분담은 다음과 같다.

| 위치 | 역할 | 갱신 시점 |
|---|---|---|
| [`reports/README.md`](reports/README.md) | **리포트 인덱스** — 진행 비교표·읽기 순서·정본/취합 구분 | 헤드라인 수치가 바뀔 때 |
| [`docs/week_N.md`](docs/) | **주차별 실험 노트** — 날짜별 진행·결과·발견·결정 + TODO 관리 | 작업할 때마다 |
| `reports/<실험>.md` | 대실험별 취합 리포트 — 변형 비교·분석·최종 결론 | 대실험 종료 시 |
| `runs/<실험>/<변형>/` | 실행 산출물 — `train.log` · `metrics.json` | 학습 실행 시. **체크포인트는 Drive 미러 보관** — 목록·sha256·복구는 [`runs/CHECKPOINTS.md`](runs/CHECKPOINTS.md), 예외로 `runs/stage_a/*/model.pt`만 git 추적(진단 스크립트가 직접 읽는다) |
| `configs/<실험>/<변형>.yaml` | 실험 설정 — `experiment`·`run_name` 키 필수 | 실험 설계 시 |

- 실험은 **대실험(experiment) / 변형(run)** 2단 구조. 변형 이름은 번호가 아니라
  무엇이 다른지 드러나는 서술형으로 짓는다 (예: `dropout0.0`, `d2-fft`).
- 실험 노트의 주차는 첫 커밋(2026-08-08) 기준 7일 단위 — week_1 = 08-08~08-14.
- 격자 스냅 등 누설 지표는 주 결과로 쓰지 않고 분리 보고한다 (§7.5).

## 12. 참고자료

- H. A. Macleod, *Thin-Film Optical Filters* — 특성행렬 정식화
- M. Born & E. Wolf, *Principles of Optics* — 다층막 간섭 이론
- [refractiveindex.info](https://refractiveindex.info) — Si / SiO₂ / Si₃N₄ 분산 데이터 (캘리브레이션 초기값)
- [대회 데이터 설명](https://dacon.io/competitions/official/235554/data)

## 라이선스

코드는 MIT. 데이터는 포함하지 않으며 데이콘 대회 규정을 따른다.

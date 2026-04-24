# =========================================================
# 0. 라이브러리
# =========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import platform
import matplotlib.font_manager as fm
import folium
import hashlib
import webbrowser
import os

warnings.filterwarnings("ignore")


# =========================================================
# 1. 한글 폰트 설정
# =========================================================
if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")
elif platform.system() == "Darwin":
    plt.rc("font", family="AppleGothic")
else:
    try:
        fm.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
        plt.rc("font", family="NanumGothic")
    except:
        pass

plt.rcParams["axes.unicode_minus"] = False


# =========================================================
# 2. 파일 읽기
#    main.py와 같은 폴더에 csv 파일을 두면 됨
# =========================================================
facility = pd.read_csv("facility.csv", encoding="cp949")
resident = pd.read_csv("resident.csv", encoding="cp949")
dementia = pd.read_csv("dementia.csv", encoding="cp949")


# =========================================================
# 3. 컬럼 찾기 함수
# =========================================================
def find_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"컬럼을 찾지 못했습니다: {candidates}\n현재 컬럼: {list(df.columns)}")
    return None


# =========================================================
# 4. 시도명 통일 함수
# =========================================================
def normalize_sido(x):
    x = str(x).strip()

    mapping = {
        "전라북도": "전북특별자치도",
        "전북특별자치도": "전북특별자치도",
        "강원도": "강원특별자치도",
        "강원특별자치도": "강원특별자치도"
    }

    return mapping.get(x, x)


# =========================================================
# 5. facility 전처리
#    예: '서울특별시 종로구 구기동' -> '서울특별시 종로구'
# =========================================================
facility_code_col = find_col(facility, ["장기요양기관코드", "기관코드"])
facility_addr_col = find_col(facility, ["시도 시군구 법정동명", "주소", "소재지", "기관주소"])

facility = facility.rename(columns={
    facility_code_col: "기관코드",
    facility_addr_col: "주소원문"
})

facility["기관코드"] = facility["기관코드"].astype(str).str.strip()
facility["주소원문"] = facility["주소원문"].astype(str).str.strip()

def extract_sido_sigungu(addr):
    if pd.isna(addr):
        return None

    parts = str(addr).strip().split()

    if len(parts) >= 2:
        return parts[0] + " " + parts[1]
    return None

facility["지역"] = facility["주소원문"].apply(extract_sido_sigungu)
facility["시도"] = facility["지역"].apply(lambda x: x.split()[0] if isinstance(x, str) and len(x.split()) >= 2 else None)
facility["시군구"] = facility["지역"].apply(lambda x: x.split()[1] if isinstance(x, str) and len(x.split()) >= 2 else None)

facility["시도"] = facility["시도"].apply(normalize_sido)
facility["지역"] = facility["시도"].astype(str).str.strip() + " " + facility["시군구"].astype(str).str.strip()

facility = facility[["기관코드", "지역", "시도", "시군구", "주소원문"]].dropna(subset=["기관코드", "지역"])


# =========================================================
# 6. resident 전처리
# =========================================================
resident_code_col = find_col(resident, ["장기요양기관코드", "기관코드"])
resident_capacity_col = find_col(resident, ["정원"])
resident_current_col = find_col(resident, ["현원"], required=False)

resident = resident.rename(columns={
    resident_code_col: "기관코드",
    resident_capacity_col: "정원"
})

if resident_current_col:
    resident = resident.rename(columns={resident_current_col: "현원"})
else:
    resident["현원"] = 0

resident["기관코드"] = resident["기관코드"].astype(str).str.strip()
resident["정원"] = pd.to_numeric(resident["정원"], errors="coerce").fillna(0)
resident["현원"] = pd.to_numeric(resident["현원"], errors="coerce").fillna(0)

resident = resident[["기관코드", "정원", "현원"]]


# =========================================================
# 7. facility + resident 병합
# =========================================================
facility_resident = facility.merge(resident, on="기관코드", how="inner")

region_capacity = facility_resident.groupby(["지역", "시도", "시군구"], as_index=False).agg(
    기관수=("기관코드", "nunique"),
    총정원=("정원", "sum"),
    총현원=("현원", "sum")
)


# =========================================================
# 8. dementia 전처리
# =========================================================
year_col = find_col(dementia, ["연도", "년도"], required=False)
sido_col = find_col(dementia, ["시도", "시도명", "광역시도"], required=False)
sigungu_col = find_col(dementia, ["시군구", "시군구명", "구군"], required=False)
gender_col = find_col(dementia, ["성별"], required=False)
age_col = find_col(dementia, ["연령별", "연령구간"], required=False)
patient_col = find_col(dementia, ["추정치매환자수", "치매환자수", "환자수", "추정환자수"])

rename_map = {patient_col: "추정치매환자수"}

if year_col:
    rename_map[year_col] = "연도"
if sido_col:
    rename_map[sido_col] = "시도"
if sigungu_col:
    rename_map[sigungu_col] = "시군구"
if gender_col:
    rename_map[gender_col] = "성별"
if age_col:
    rename_map[age_col] = "연령별"

dementia = dementia.rename(columns=rename_map)

for c in ["시도", "시군구", "성별", "연령별"]:
    if c in dementia.columns:
        dementia[c] = dementia[c].astype(str).str.strip()

dementia["시도"] = dementia["시도"].apply(normalize_sido)

if "성별" in dementia.columns:
    dementia = dementia[dementia["성별"] == "전체"].copy()

dementia["연령별"] = dementia["연령별"].astype(str).str.strip().str.replace(" ", "")

target_ages = [
    "60~64세",
    "65~69세",
    "70~74세",
    "75~79세",
    "80~84세",
    "85세이상"
]

dementia = dementia[dementia["연령별"].isin(target_ages)].copy()

dementia["추정치매환자수"] = pd.to_numeric(dementia["추정치매환자수"], errors="coerce").fillna(0)

remove_words = ["전국", "전체", "합계", "소계"]
dementia = dementia[~dementia["시도"].isin(remove_words)].copy()
dementia = dementia[~dementia["시군구"].isin(remove_words)].copy()

dementia = dementia[dementia["시군구"] != dementia["시도"]].copy()
# =========================================================
# 시군구를 '시/군' 단위로 통일 (구 제거)
# =========================================================
dementia["시군구"] = dementia["시군구"].str.replace(r"(.*시).*", r"\1", regex=True)

# '시', '군'만 남기기
dementia = dementia[
    dementia["시군구"].str.endswith(("시", "군"))
].copy()

if "연도" in dementia.columns:
    dementia["연도"] = pd.to_numeric(dementia["연도"], errors="coerce")
    latest_year = int(dementia["연도"].dropna().max())
    dementia = dementia[dementia["연도"] == latest_year].copy()
    print(f"치매 데이터 사용 연도: {latest_year}")

dementia["지역"] = dementia["시도"].astype(str).str.strip() + " " + dementia["시군구"].astype(str).str.strip()

dementia_region = dementia.groupby(["지역", "시도", "시군구"], as_index=False).agg(
    치매환자수=("추정치매환자수", "sum")
)


# =========================================================
# 9. 최종 병합
# =========================================================
final = dementia_region.merge(
    region_capacity,
    on=["지역", "시도", "시군구"],
    how="left"
).fillna(0)

final["총정원"] = pd.to_numeric(final["총정원"], errors="coerce").fillna(0)
final["총현원"] = pd.to_numeric(final["총현원"], errors="coerce").fillna(0)
final["치매환자수"] = pd.to_numeric(final["치매환자수"], errors="coerce").fillna(0)

final["정원차이"] = final["총정원"] - final["치매환자수"]
final["부족인원"] = final["치매환자수"] - final["총정원"]
final["판단"] = np.where(final["총정원"] >= final["치매환자수"], "적당", "부족")
final["정원대비비율"] = (final["총정원"] / final["치매환자수"].replace(0, np.nan)).round(2)

final = final.sort_values(["판단", "부족인원"], ascending=[True, False]).reset_index(drop=True)


# =========================================================
# 10. 그래프
# =========================================================
shortage = final[final["판단"] == "부족"].copy()
shortage_top20 = shortage.sort_values("부족인원", ascending=False).head(20)

plt.figure(figsize=(18, 7))
plt.bar(shortage_top20["지역"], shortage_top20["부족인원"])
plt.title("요양시설이 부족한 지역 TOP 20")
plt.xlabel("지역")
plt.ylabel("부족 인원 수(치매환자수 - 총정원)")
plt.xticks(rotation=75)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

compare_top20 = final.sort_values("치매환자수", ascending=False).head(20)

plt.figure(figsize=(18, 7))
plt.plot(compare_top20["지역"], compare_top20["치매환자수"], marker="o", label="치매환자수")
plt.plot(compare_top20["지역"], compare_top20["총정원"], marker="s", label="요양기관 총정원")
plt.title("지역별 치매환자수와 요양기관 총정원 비교")
plt.xlabel("지역")
plt.ylabel("인원 수")
plt.xticks(rotation=75)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# =========================================================
# 11. 지도 시각화
# =========================================================
sido_centers = {
    "서울특별시": (37.5665, 126.9780),
    "부산광역시": (35.1796, 129.0756),
    "대구광역시": (35.8714, 128.6014),
    "인천광역시": (37.4563, 126.7052),
    "광주광역시": (35.1595, 126.8526),
    "대전광역시": (36.3504, 127.3845),
    "울산광역시": (35.5384, 129.3114),
    "세종특별자치시": (36.4800, 127.2890),
    "경기도": (37.4138, 127.5183),
    "강원특별자치도": (37.8228, 128.1555),
    "충청북도": (36.6357, 127.4917),
    "충청남도": (36.6588, 126.6728),
    "전북특별자치도": (35.7175, 127.1530),
    "전라남도": (34.8679, 126.9910),
    "경상북도": (36.4919, 128.8889),
    "경상남도": (35.4606, 128.2132),
    "제주특별자치도": (33.4996, 126.5312)
}

def get_jitter(region_name, scale=0.18):
    h = hashlib.md5(region_name.encode("utf-8")).hexdigest()
    a = int(h[:8], 16)
    b = int(h[8:16], 16)

    lat_jitter = ((a % 1000) / 1000 - 0.5) * scale
    lon_jitter = ((b % 1000) / 1000 - 0.5) * scale
    return lat_jitter, lon_jitter

korea_map = folium.Map(location=[36.35, 127.8], zoom_start=7)

for _, row in final.iterrows():
    sido = row["시도"]
    region = row["지역"]

    if sido not in sido_centers:
        continue

    base_lat, base_lon = sido_centers[sido]
    lat_jitter, lon_jitter = get_jitter(region)

    lat = base_lat + lat_jitter
    lon = base_lon + lon_jitter

    color = "red" if row["판단"] == "부족" else "blue"

    shortage_value = max(0, row["부족인원"])
    radius = 4 + min(shortage_value / 500, 12)

    popup_text = f"""
    <b>{row['지역']}</b><br>
    치매환자수: {int(row['치매환자수']):,}명<br>
    총정원: {int(row['총정원']):,}명<br>
    기관수: {int(row['기관수']):,}개<br>
    정원차이: {int(row['정원차이']):,}명<br>
    판단: {row['판단']}
    """

    folium.CircleMarker(
        location=[lat, lon],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        popup=folium.Popup(popup_text, max_width=300),
        tooltip=row["지역"]
    ).add_to(korea_map)

legend_html = """
<div style="
    position: fixed;
    bottom: 50px; left: 50px; width: 180px; height: 90px;
    background-color: white; z-index:9999; font-size:14px;
    border:2px solid grey; padding:10px;
">
<b>요양시설 적정 여부</b><br>
<span style="color:red;">●</span> 부족<br>
<span style="color:blue;">●</span> 적당
</div>
"""
korea_map.get_root().html.add_child(folium.Element(legend_html))


# =========================================================
# 12. 결과 저장
# =========================================================
final.to_csv("지역별_치매환자수_요양기관정원_비교_통일본.csv", index=False, encoding="utf-8-sig")

map_path = os.path.abspath("치매_요양시설_지도.html")
korea_map.save(map_path)

print("CSV 저장 완료: 지역별_치매환자수_요양기관정원_비교_통일본.csv")
print(f"지도 저장 완료: {map_path}")


# =========================================================
# 13. 지도 자동 열기
# =========================================================
webbrowser.open("file://" + map_path)
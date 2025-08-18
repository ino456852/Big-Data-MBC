import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import numpy as np
import os
import platform

from seaborn import color_palette

# ===============================================================
# 1. 한글 폰트 설정 및 폴더 생성
# ===============================================================

# 운영체제에 맞는 한글 폰트 설정
system_name = platform.system()
if system_name == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif system_name == 'Darwin': # Mac OS
    plt.rc('font', family='AppleGothic')
elif system_name == 'Linux':
    # Google Colab 또는 Linux 환경에서는 나눔고딕을 설치해야 할 수 있습니다.
    # !sudo apt-get install -y fonts-nanum
    # !sudo fc-cache -fv
    # !rm ~/.cache/matplotlib -rf
    # 위 코드 실행 후 '런타임 다시 시작'을 해야 폰트가 적용됩니다.
    plt.rc('font', family='NanumGothic')

# 마이너스 부호 깨짐 방지
plt.rc('axes', unicode_minus=False)

# '시각화' 폴더 생성
folder_name = '시각화'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"'{folder_name}' 폴더를 생성했습니다.")


df = sns.load_dataset('diamonds')
df.info()
print('x=',df['x'], 'y=',df['y'], 'z=',df['z'])

matplotlib.use('TkAgg', force=True)

# 1. Scatter Plot (산점도)
# '컷팅' 컬럼 이름 변경
df.rename(columns={'cut': '컷팅'})
# '컷팅' 컬럼의 범주 값을 한글로 변경
df['컷팅'].replace({
    'Ideal': '훌륭함',
    'Premium': '매우 좋음',
    'Very Good': '좋음',
    'Good': '보통',
    'Fair': '미흡'
})
color_palette = ({
    '훌륭함':'#f25f7a',
    '매우 좋음':'#f27e91',
    '좋음':'#f4a6b3',
    '보통':'#f2c6b6',
    '미흡':'#f0e1d2'
})

carrat_palette = ['red', '#f4a261', '#e9c46a', '#2a9d8f', '#264653']
print(df['컷팅'].cat.categories)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='carat', y='price', hue='컷팅', palette=carrat_palette, alpha=0.5)
plt.title('캐럿과 가격의 관계 (cut 등급 별)', fontsize=20)
plt.xlabel('캐럿 (Carat)', fontsize=15)
plt.ylabel('가격 (Price)', fontsize=15)
plt.xlim(0, 3)
plt.ylim(0,15000)
plt.savefig(os.path.join(folder_name, '1_scatter_plot.png'))
plt.show()




# 2. Bar Plot (막대 그래프)
mean = df[df['컷팅'] == '미흡']['price'].mean()
plt.figure(figsize=(10, 6))
plt.legend()
ax = sns.barplot(x='컷팅', y='price', data=df, palette=color_palette)
plt.title('컷 품질에 따른 평균 가격', fontsize=20)
plt.xlabel('컷 (Cut)', fontsize=15)
plt.ylabel('평균 가격 (Price)', fontsize=15)
ax.axhline(y=mean, color='red', linestyle='--', linewidth=2, label=mean)
plt.savefig(os.path.join(folder_name, '2_bar_plot.png'))
plt.show()



# 3. Horizontal Bar Plot (수평 막대 그래프)
plt.figure(figsize=(10, 6))
color_order = df.groupby('color')['price'].mean().sort_values().index
sns.barplot(x='price', y='color', data=df, orient='h',
            order=color_order, palette='Reds')
plt.title('색상에 따른 평균 가격', fontsize=20)
plt.xlabel('평균 가격 (Price)', fontsize=15)
plt.ylabel('색상 (Color)', fontsize=15)
plt.savefig(os.path.join(folder_name, '3_barh_plot.png'))
plt.show()



# 4. Line Chart (선 그래프)
price_by_carat = df.groupby(pd.cut(df['carat'], bins=np.arange(0, 6, 0.5)))['price'].mean()
plt.figure(figsize=(10, 6))
price_by_carat.plot(kind='line', marker='o')
plt.title('캐럿 크기에 따른 평균 가격 변화', fontsize=20)
plt.xlabel('캐럿 (Carat)', fontsize=15)
plt.ylabel('평균 가격 (Price)', fontsize=15)
plt.grid(True)
plt.savefig(os.path.join(folder_name, '4_line_chart.png'))
plt.show()

# 5. Area Plot (영역 그래프)
plt.figure(figsize=(10, 6))
price_by_carat.plot(kind='area', alpha=0.5)
plt.title('캐럿 크기에 따른 평균 가격 변화 (영역 그래프)', fontsize=20)
plt.xlabel('캐럿 (Carat)', fontsize=15)
plt.ylabel('평균 가격 (Price)', fontsize=15)
plt.grid(True)
plt.savefig(os.path.join(folder_name, '5_area_plot.png'))
plt.show()

# 6. Histogram (히스토그램)
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=50, kde=True)
plt.title('다이아몬드 가격 분포', fontsize=20)
plt.xlabel('가격 (Price)', fontsize=15)
plt.ylabel('빈도 (Frequency)', fontsize=15)
plt.savefig(os.path.join(folder_name, '6_histogram.png'))
plt.show()

# 7. Pie Chart (파이 차트)
cut_counts = df['cut'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(cut_counts, labels=cut_counts.index, autopct='%1.1f%%', startangle=140,
        colors=sns.color_palette('pastel'),
        textprops={'fontsize': 14})
plt.title('컷 품질의 비율', fontsize=15)
plt.ylabel('')
plt.savefig(os.path.join(folder_name, '7_pie_chart.png'))
plt.show()

# 8. Box Plot (박스 플롯)
plt.figure(figsize=(10, 6))
sns.boxplot(x='컷팅', y='price', data=df, palette=color_palette)
plt.title('컷 품질에 따른 가격 분포', fontsize=20)
plt.xlabel('컷 (Cut)', fontsize=15)
plt.ylabel('가격 (Price)', fontsize=15)
plt.savefig(os.path.join(folder_name, '8_box_plot.png'))
plt.show()

# 9. 3D Graph (3D 그래프) - Plotly
fig = px.scatter_3d(df.sample(2000), x='carat', y='price', z='depth', color='cut', title='캐럿, 가격, 깊이의 3D 관계')
fig.write_html(os.path.join(folder_name, "9_3d_scatter_plot.html"))

# 10. lmplot (회귀선 플롯)
lm = sns.lmplot(x='carat', y='price', hue='컷팅', data=df, height=7, aspect=1.2, scatter_kws={'alpha':0.1})
lm.fig.suptitle('캐럿과 가격의 선형 회귀 관계 (컷 품질별)', y=1.02, fontsize=15)
plt.savefig(os.path.join(folder_name, '10_lmplot.png'))
plt.show()

# 11. Count Plot (개수 플롯)
plt.figure(figsize=(10, 6))
sns.countplot(x='cut', data=df, order=cut_order)
plt.title('컷 품질별 다이아몬드 개수', fontsize=15)
plt.xlabel('컷 (Cut)')
plt.ylabel('개수 (Count)')
plt.savefig(os.path.join(folder_name, '11_countplot.png'))
plt.close()

# 12. Heatmap (히트맵)
numerical_df = df.select_dtypes(include=np.number)
correlation_matrix = numerical_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('수치형 변수 간의 상관관계', fontsize=15)
plt.savefig(os.path.join(folder_name, '12_heatmap.png'))
plt.close()

# 13. Joint Plot (조인트 플롯)
jp = sns.jointplot(x='carat', y='price', data=df, kind='scatter', height=8, joint_kws={'alpha':0.1})
jp.fig.suptitle('캐럿과 가격의 관계 및 분포', y=1.02, fontsize=15)
plt.savefig(os.path.join(folder_name, '13_joint_plot.png'))
plt.close()

# 14. KDE Plot (커널 밀도 추정 플롯)
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='price', hue='cut', fill=True, common_norm=False, palette="crest", alpha=.5, linewidth=0)
plt.title('컷 품질에 따른 가격의 밀도 분포', fontsize=15)
plt.xlabel('가격 (Price)')
plt.ylabel('밀도 (Density)')
plt.savefig(os.path.join(folder_name, '14_kdeplot.png'))
plt.close()

# 15. Pair Plot (페어 플롯)
# 계산 시간이 오래 걸릴 수 있어 200개 데이터만 샘플링하여 사용합니다.
sampled_df = df.sample(n=200)
pp = sns.pairplot(sampled_df, hue='cut')
pp.fig.suptitle('다이아몬드 변수 간의 관계 (샘플링 데이터)', y=1.02, fontsize=15)
plt.savefig(os.path.join(folder_name, '15_pairplot.png'))
plt.close()

# 16. Relational Plot (replot)
rp = sns.relplot(x='carat', y='price', hue='color', size='depth',
            sizes=(10, 200), alpha=.5, palette="muted",
            height=6, data=df)
rp.fig.suptitle('캐럿, 가격, 색상, 깊이의 다차원 관계', y=1.02, fontsize=15)
plt.savefig(os.path.join(folder_name, '16_relplot.png'))
plt.close()

# 17. Rug Plot (러그 플롯)
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='price', fill=True)
sns.rugplot(data=df, x='price', height=0.05)
plt.title('가격 분포와 Rug Plot', fontsize=15)
plt.xlabel('가격 (Price)')
plt.ylabel('밀도 (Density)')
plt.savefig(os.path.join(folder_name, '17_rugplot.png'))
plt.close()

# 18. Violin Plot (바이올린 플롯)
plt.figure(figsize=(10, 6))
sns.violinplot(x='cut', y='price', data=df, order=cut_order)
plt.title('컷 품질에 따른 가격의 분포 (Violin Plot)', fontsize=15)
plt.xlabel('컷 (Cut)')
plt.ylabel('가격 (Price)')
plt.savefig(os.path.join(folder_name, '18_violinplot.png'))
plt.close()

# 19. Interactive Plot (인터랙티브 플롯) - Plotly
fig = px.scatter(df.sample(1000), x="carat", y="price", color="cut",
                 size='depth', hover_data=['clarity', 'color'],
                 title='캐럿과 가격의 인터랙티브 관계')
fig.write_html(os.path.join(folder_name, "19_interactive_scatter_plot.html"))


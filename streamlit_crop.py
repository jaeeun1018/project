import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# 페이지 설정
st.set_page_config(
    page_title='농작물 생산량 예측',
    page_icon='🌾',
    layout='centered'
)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# 모델 및 전처리 객체 불러오기
model = joblib.load('crop_model.joblib')
encoder = joblib.load('encoder.joblib')
scaler = joblib.load('scaler.joblib')


st.title('농작물 생산량 예측')
st.write('기상 및 농업 데이터를 입력하여 농작물 생산량을 예측합니다.')


# 학습에 사용된 작물과 지역 목록
category_map = {
    col: list(categories)
    for col, categories in zip(
        encoder.feature_names_in_,
        encoder.categories_
    )
}

crop_options = sorted(category_map['Crop'])
state_options = sorted(category_map['State'])


# 예측 / 평가 탭
tab1, tab2 = st.tabs(['예측하기', '📊 평가시각화'])



with tab1:

    st.subheader('농작물 및 환경 정보 입력')

    crop = st.selectbox('작물 종류', crop_options)
    state = st.selectbox('지역', state_options)

    area = st.number_input(
        '재배 면적 (ha)',
        min_value=0.0,
        value=0.0,
        step=100.0
    )

    rainfall = st.number_input(
        '연간 강수량 (mm)',
        min_value=0.0,
        value=0.0,
        step=10.0
    )

    fertilizer = st.number_input(
        '비료 사용량 (kg)',
        min_value=0.0,
        value=0.0,
        step=100.0
    )

    pesticide = st.number_input(
        '농약 사용량 (kg)',
        min_value=0.0,
        value=0.0,
        step=100.0
    )

    avg_temp = st.number_input(
        '평균 기온 (℃)',
        value=0.0,
        step=0.1
    )

    max_temp = st.number_input(
        '최고 기온 (℃)',
        value=0.0,
        step=0.1
    )

    min_temp = st.number_input(
        '최저 기온 (℃)',
        value=0.0,
        step=0.1
    )


    if st.button(
        '생산량 예측',
        type='primary',
        use_container_width=True
    ):

        # 재배 면적이 0이면 생산량도 0으로 처리
        if area == 0:
            prediction = 0.0

        # 기온 입력값 확인
        elif min_temp > max_temp:
            st.error('최저 기온은 최고 기온보다 높을 수 없습니다.')
            st.stop()

        elif not min_temp <= avg_temp <= max_temp:
            st.error('평균 기온은 최저 기온과 최고 기온 사이의 값을 입력해주세요.')
            st.stop()

        else:
            input_data = pd.DataFrame([{
                'Crop': crop,
                'State': state,
                'Area': area,
                'Annual_Rainfall': rainfall,
                'Fertilizer': fertilizer,
                'Pesticide': pesticide,
                'Avg_Temperature': avg_temp,
                'Max_Temperature': max_temp,
                'Min_Temperature': min_temp
            }])

            categorical_cols = ['Crop', 'State']

            numeric_cols = [
                'Area',
                'Annual_Rainfall',
                'Fertilizer',
                'Pesticide',
                'Avg_Temperature',
                'Max_Temperature',
                'Min_Temperature'
            ]

            # 범주형 데이터 인코딩
            input_cat = encoder.transform(
                input_data[categorical_cols]
            )

            encoded_cat_cols = encoder.get_feature_names_out(
                categorical_cols
            )

            input_cat_df = pd.DataFrame(
                input_cat,
                columns=encoded_cat_cols
            )

            # 수치형과 범주형 데이터 결합
            input_encoded = pd.concat(
                [
                    input_data[numeric_cols].reset_index(drop=True),
                    input_cat_df.reset_index(drop=True)
                ],
                axis=1
            )

            # 학습 당시 컬럼 순서에 맞춤
            if hasattr(scaler, 'feature_names_in_'):
                input_encoded = input_encoded.reindex(
                    columns=scaler.feature_names_in_,
                    fill_value=0
                )

            input_scaled = scaler.transform(input_encoded)

            prediction = model.predict(input_scaled)[0]


        st.metric(
            label='🌾 예상 농작물 생산량',
            value=f'{prediction:,.2f} 톤'
        )



with tab2:

    st.subheader('최종 모델 성능 정보')
    st.write('**최종 모델 : Random Forest Regressor**')

    col1, col2, col3 = st.columns(3)

    col1.metric('R²', '0.977')
    col2.metric('MAE', '2,249,780')
    col3.metric('RMSE', '40,537,911')

    st.divider()

    st.subheader('주요 변수 중요도')


    # 모델 학습에 사용된 피처 이름
    if hasattr(scaler, 'feature_names_in_'):
        feature_names = list(scaler.feature_names_in_)

    else:
        numeric_cols = [
            'Area',
            'Annual_Rainfall',
            'Fertilizer',
            'Pesticide',
            'Avg_Temperature',
            'Max_Temperature',
            'Min_Temperature'
        ]

        encoded_cols = encoder.get_feature_names_out(
            ['Crop', 'State']
        ).tolist()

        feature_names = numeric_cols + encoded_cols


    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    })


    # 원핫인코딩된 피처를 원래 변수 단위로 묶음
    def get_feature_group(feature):

        if feature.startswith('Crop_'):
            return '작물 종류'

        if feature.startswith('State_'):
            return '지역'

        feature_names_kr = {
            'Area': '재배 면적',
            'Annual_Rainfall': '연간 강수량',
            'Fertilizer': '비료 사용량',
            'Pesticide': '농약 사용량',
            'Avg_Temperature': '평균 기온',
            'Max_Temperature': '최고 기온',
            'Min_Temperature': '최저 기온'
        }

        return feature_names_kr.get(feature, feature)


    importance_df['Feature_Group'] = (
        importance_df['Feature'].apply(get_feature_group)
    )

    grouped_importance = (
        importance_df
        .groupby('Feature_Group', as_index=False)['Importance']
        .sum()
        .sort_values('Importance', ascending=False)
    )


    # 피처 중요도 그래프
    plot_df = grouped_importance.sort_values(
        'Importance',
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        plot_df['Feature_Group'],
        plot_df['Importance']
    )

    ax.set_title('Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')

    plt.tight_layout()
    st.pyplot(fig)


    st.caption(
        '변수 중요도가 높을수록 농작물 생산량 예측에 '
        '상대적으로 큰 영향을 미친 변수입니다.'
    )

    st.dataframe(
        grouped_importance,
        use_container_width=True,
        hide_index=True
    )
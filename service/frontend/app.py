import os
from typing import Optional

import pandas as pd
import requests
import streamlit as st


# Конфигурация
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Advisory Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .rec-card {
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
        background: #f8fafc;
    }
    .tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.78rem;
        font-weight: 500;
        margin-right: 4px;
    }
    .tag-blue   { background:#dbeafe; color:#1d4ed8; }
    .tag-green  { background:#dcfce7; color:#15803d; }
    .tag-orange { background:#ffedd5; color:#c2410c; }
    .tag-gray   { background:#f1f5f9; color:#475569; }
</style>
""", unsafe_allow_html=True)


# Маппинги
JUSTIFICATION_MAP = {
    "risk_profile": "Соответствует риск-профилю клиента",
    "similar_clients": "Покупали клиенты с похожим портфелем",
    "portfolio_complement": "Дополнит текущую структуру портфеля клиента",
    "popularity": "Популярно среди клиентов банка",
    "item_cf": "Похоже на активы из портфеля клиента",
    "user_cf": "Покупали клиенты с похожим профилем",
    "portfolio": "Дополнит текущую структуру портфеля клиента",
    "popular_sector": "Популярно среди клиентов аналогичного сектора",
    "popular_category": "Популярно среди клиентов с похожими предпочтениями",
}

INVESTMENT_CAPACITY_MAP = {
    "CAP_LT30K": "< 30 000 €",
    "CAP_30K_80K": "30 000 – 80 000 €",
    "CAP_80K_300K": "80 000 – 300 000 €",
    "CAP_GT300K": "> 300 000 €",
    "Not_Available": "Не указано",
    "Predicted_CAP_LT30K": "< 30 000 € *(прогноз)*",
    "Predicted_CAP_30K_80K": "30 000 – 80 000 € *(прогноз)*",
    "Predicted_CAP_80K_300K": "80 000 – 300 000 € *(прогноз)*",
    "Predicted_CAP_GT300K": "> 300 000 € *(прогноз)*",
}

SEGMENT_ICON = {"warm": "🟩", "dormant": "🟧"}
REC_COLORS = ["#2563EB", "#16A34A", "#D97706"]


def api_get(path: str, params: dict = None) -> Optional[dict]:
    try:
        resp = requests.get(f"{API_URL}{path}", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Не удалось подключиться к API ({API_URL}). Проверьте что бэкенд запущен.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"Ошибка API {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Неожиданная ошибка: {e}")
        return None


@st.cache_data(ttl=300)
def get_snapshot_date() -> Optional[str]:
    data = api_get("/api/snapshot-date")
    return data.get("snapshot_date") if data else None


@st.cache_data(ttl=300)
def get_advisors() -> list[dict]:
    data = api_get("/api/advisors")
    return data.get("items", []) if data else []


@st.cache_data(ttl=300)
def get_stats() -> dict:
    data = api_get("/api/clients/stats")
    return data if data else {"max_days": 730, "total_hot": 0}


# Инициализация session_state
def _init_state():
    defaults = {
        "page": "list",
        "selected_client": None,
        "filter_segment": "Все",
        "filter_score": (0.0, 1.0),
        "filter_days": None,
        "filter_advisor": "Все советники",
        "list_page": 1,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# Экран 1: Список клиентов
def show_client_list():
    snapshot_date = get_snapshot_date()
    stats = get_stats()
    max_days = stats.get("max_days", 730)

    if st.session_state["filter_days"] is None:
        st.session_state["filter_days"] = (0, max_days)

    date_label = f" — данные на {snapshot_date}" if snapshot_date else ""
    st.title(f"Клиенты для обзвона{date_label}")

    with st.sidebar:
        st.header("Фильтры")

        search_id = st.text_input(
            "Поиск по ID клиента",
            value="",
            placeholder="Введите ID клиента",
        )

        # Советник
        advisors = get_advisors()
        advisor_opts = ["Все советники"] + [a["advisor_name"] for a in advisors]
        advisor_sel = st.selectbox(
            "Советник",
            advisor_opts,
            index=advisor_opts.index(st.session_state["filter_advisor"])
            if st.session_state["filter_advisor"] in advisor_opts else 0,
        )

        # Сегмент
        seg_opts = ["Все", "Warm", "Dormant"]
        segment  = st.selectbox(
            "Сегмент",
            seg_opts,
            index=seg_opts.index(st.session_state["filter_segment"])
            if st.session_state["filter_segment"] in seg_opts else 0,
        )

        # Propensity score
        score_range = st.slider(
            "Propensity Score",
            min_value=0.0, max_value=1.0,
            value=st.session_state["filter_score"],
            step=0.01,
        )


        # Дней без покупки
        days_val = st.session_state["filter_days"]
        if not isinstance(days_val, tuple):
            days_val = (0, max_days)
        days_range = st.slider(
            "Дней без покупки",
            min_value=0, max_value=max_days,
            value=days_val,
            step=1,
        )

        col_apply, col_reset = st.columns(2)
        with col_apply:
            apply = st.button("Применить", use_container_width=True)
        with col_reset:
            reset = st.button("Сбросить", use_container_width=True)

        if apply:
            st.session_state.update({
                "filter_advisor": advisor_sel,
                "filter_segment": segment,
                "filter_score": score_range,
                "filter_days": days_range,
                "list_page": 1,
            })
            st.rerun()

        if reset:
            st.session_state.update({
                "filter_advisor": "Все советники",
                "filter_segment": "Все",
                "filter_score": (0.0, 1.0),
                "filter_days": (0, max_days),
                "list_page": 1,
            })
            st.rerun()

    cur_days  = st.session_state["filter_days"]
    if not isinstance(cur_days, tuple):
        cur_days = (0, max_days)

    params = {
        "score_min": st.session_state["filter_score"][0],
        "score_max": st.session_state["filter_score"][1],
        "days_min": cur_days[0],
        "days_max": cur_days[1],
        "page": st.session_state["list_page"],
        "page_size": 50,
    }
    if search_id and search_id.strip():
        params["search_id"] = search_id.strip()
    if st.session_state["filter_segment"] != "Все":
        params["segment"] = st.session_state["filter_segment"].lower()
    if st.session_state["filter_advisor"] != "Все советники":
        params["advisor_name"] = st.session_state["filter_advisor"]

    with st.spinner("Загрузка данных..."):
        data = api_get("/api/clients", params=params)

    if data is None:
        return

    items = data.get("items", [])
    total = data.get("total", 0)
    cur_page = data.get("page", 1)
    page_size = data.get("page_size", 50)
    total_pages = max(1, (total + page_size - 1) // page_size)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Клиентов (фильтр)", f"{total:,}")
    m2.metric("🟩 Warm",    sum(1 for i in items if i.get("segment") == "warm"))
    m3.metric("🟧 Dormant", sum(1 for i in items if i.get("segment") == "dormant"))
    if items:
        avg = sum(i.get("propensity_score", 0) for i in items) / len(items)
        m4.metric("Средний скор", f"{avg:.3f}")

    st.divider()

    if not items:
        advisor_sel = st.session_state["filter_advisor"]
        cur_score = st.session_state["filter_score"]
        cur_seg = st.session_state["filter_segment"]
        cur_days = st.session_state["filter_days"] or (0, max_days)
        filters_active = (
            cur_seg != "Все"
            or cur_score != (0.0, 1.0)
            or cur_days != (0, max_days)
        )

        if advisor_sel != "Все советники" and not filters_active:
            st.info(
                f"На {snapshot_date} у советника **{advisor_sel}** "
                f"нет клиентов для звонка. "
                f"Все клиенты охвачены или не прошли скоринг на эту дату."
            )
        else:
            st.info(
                "По выбранным фильтрам клиентов не найдено. "
                "Попробуйте изменить параметры фильтрации."
            )
        return

    hdr = st.columns([0.5, 1.8, 0.9, 0.9, 1.1, 1.2, 0.9])
    for col, lbl in zip(hdr, ["Ранг", "Клиент ID", "Сегмент", "Score", "Дней без покупки", "Последний актив", ""]):
        col.markdown(f"**{lbl}**")
    st.markdown("<hr style='margin:4px 0 8px 0'>", unsafe_allow_html=True)

    for item in items:
        seg   = item.get("segment", "")
        score = item.get("propensity_score", 0)
        days  = item.get("days_since_last_buy")

        cols = st.columns([0.5, 1.8, 0.9, 0.9, 1.1, 1.2, 0.9])
        cols[0].write(f"#{item.get('rank', '?')}")
        cols[1].write(item.get("customer_id", ""))
        cols[2].write(f"{SEGMENT_ICON.get(seg, '⚪')} {seg.capitalize()}")
        cols[3].write(f"{score:.3f}")
        cols[4].write(f"{days} дн." if days is not None else "—")
        cols[5].write(item.get("last_asset_category") or "—")

        if cols[6].button("Карточка →", key=f"btn_{item['customer_id']}"):
            st.session_state["selected_client"] = item["customer_id"]
            st.session_state["page"]            = "card"
            st.rerun()

    st.divider()

    p1, p2, p3 = st.columns([1, 2, 1])
    with p1:
        if cur_page > 1 and st.button("← Назад"):
            st.session_state["list_page"] = cur_page - 1
            st.rerun()
    with p2:
        st.markdown(
            f"<div style='text-align:center;padding-top:6px'>Страница {cur_page} из {total_pages}</div>",
            unsafe_allow_html=True,
        )
    with p3:
        if cur_page < total_pages and st.button("Вперёд →"):
            st.session_state["list_page"] = cur_page + 1
            st.rerun()


# Экран 2: Карточка клиента
def show_client_card(customer_id: str):
    if st.button("← Назад к списку"):
        st.session_state["page"]           = "list"
        st.session_state["selected_client"] = None
        st.rerun()

    with st.spinner(f"Загрузка карточки {customer_id}..."):
        data = api_get(f"/api/clients/{customer_id}")

    if data is None:
        return

    profile = data.get("profile", {})
    portfolio = data.get("portfolio", {})
    recs = data.get("recommendations", [])
    recent_tx = data.get("recent_transactions", [])

    seg = profile.get("segment", "")
    seg_label = f"{SEGMENT_ICON.get(seg, '⚪')} {seg.capitalize()}"

    st.title(f"Клиент {customer_id}   {seg_label}")

    left_col, right_col = st.columns([4, 6])

    
    with left_col:

        # Профиль
        st.subheader("Профиль")

        capacity_raw = profile.get("investment_capacity") or "Not_Available"
        capacity_display = INVESTMENT_CAPACITY_MAP.get(capacity_raw, capacity_raw)

        risk_raw = profile.get("risk_level") or "—"
        if risk_raw.startswith("Predicted_"):
            risk_display = risk_raw.replace("Predicted_", "") + " *(прогноз)*"
        else:
            risk_display = risk_raw

        info_rows = [
            ("Тип клиента", profile.get("customer_type") or "—"),
            ("Риск-профиль", risk_display),
            ("Инвест. ёмкость", capacity_display),
            ("Дней в системе", f"{profile.get('days_in_system')} дн." if profile.get("days_in_system") is not None else "—"),
            ("Дней без покупки", f"{profile.get('days_since_last_buy')} дн." if profile.get("days_since_last_buy") is not None else "—"),
        ]

        for label, value in info_rows:
            c1, c2 = st.columns([1, 1])
            c1.markdown(f"**{label}**")

            if "*(прогноз)*" in str(value):
                c2.markdown(value)
            else:
                safe = str(value).replace(">", "\\>").replace("<", "\\<")
                c2.markdown(safe)

        st.divider()
        st.metric(
            "Propensity Score",
            f"{profile.get('propensity_score', 0):.4f}",
            help="Вероятность покупки в ближайшие 30 дней (модель CatBoost)",
        )

        st.divider()

        # Активность клиента
        st.subheader("Инвестиционная активность")

        n_unique = portfolio.get("n_unique_assets", 0)
        n_buy = portfolio.get("n_buy_transactions", 0)
        first_d = portfolio.get("first_buy_date")
        last_d = portfolio.get("last_buy_date")
        top_cat = portfolio.get("top_category")
        top_sec = portfolio.get("top_sector")

        if n_buy == 0:
            st.info("История покупок отсутствует.")
        else:
            act_rows = [
                ("Покупок всего", str(n_buy)),
                ("Уникальных активов", str(n_unique)),
                ("Первая покупка", first_d or "—"),
                ("Последняя покупка", last_d  or "—"),
                ("Основная категория", top_cat or "—"),
                ("Основной сектор", top_sec or "—"),
            ]
            for label, value in act_rows:
                c1, c2 = st.columns([1, 1])
                c1.markdown(f"**{label}**")
                c2.write(value)


    with right_col:

        # Рекомендации
        st.subheader("Топ-3 рекомендации")

        if not recs:
            st.info(
                "Рекомендации для этого клиента не сформированы. "
                "Возможные причины: недостаточно истории транзакций "
                "или клиент не прошёл фильтр риск-профиля."
            )
        else:
            justifications = [r.get("justification") for r in recs]
            all_same_just  = len(set(justifications)) == 1 and justifications[0] is not None

            if all_same_just:
                j = justifications[0]
                jtext = JUSTIFICATION_MAP.get(j, j)
                st.caption(f"**Основание для всех рекомендаций:** {jtext}")

            for rec in recs:
                rank = rec.get("rank", 1)
                color = REC_COLORS[min(rank - 1, 2)]
                isin = rec.get("isin") or "—"
                name = rec.get("asset_name") or isin
                category = rec.get("category")
                sub_cat = rec.get("asset_sub_category")
                sector = rec.get("sector")
                outside = rec.get("outside_hist", False)
                j = rec.get("justification", "")
                jtext = rec.get("justification_text") or JUSTIFICATION_MAP.get(j, j)

                st.markdown(
                    f'<div style="border-left:4px solid {color};background:#f8fafc;'
                    f'border-radius:6px;padding:10px 16px;margin-bottom:4px">'
                    f'<b>{rank}. {name}</b></div>',
                    unsafe_allow_html=True,
                )

                tag_parts = []
                if category:
                    tag_parts.append(f'<span class="tag tag-blue">Категория: {category}</span>')
                if sub_cat and sub_cat != category:
                    tag_parts.append(f'<span class="tag tag-gray">Подкатегория: {sub_cat}</span>')
                if sector:
                    tag_parts.append(f'<span class="tag tag-green">Сектор: {sector}</span>')
                if outside:
                    tag_parts.append('<span class="tag tag-orange">Новая категория для клиента</span>')

                if tag_parts:
                    st.markdown(" ".join(tag_parts), unsafe_allow_html=True)

                # ISIN + обоснование
                caption_parts = [f"ISIN: `{isin}`"]
                if not all_same_just:
                    caption_parts.append(f"{jtext}")
                st.caption("  |  ".join(caption_parts))

                st.markdown("<hr style='margin:2px 0 6px 0; border:none; border-top:1px solid #e2e8f0'>", unsafe_allow_html=True)

        # История транзакций
        st.subheader("Последние транзакции")

        if not recent_tx:
            st.info("История транзакций отсутствует.")
        else:
            tx_df = pd.DataFrame(recent_tx)

            if "total_value" in tx_df.columns:
                tx_df["total_value"] = tx_df["total_value"].apply(
                    lambda x: f"{x:,.0f} €" if x is not None else "—"
                )
            if "transaction_type" in tx_df.columns:
                tx_df["transaction_type"] = tx_df["transaction_type"].map(
                    {"Buy": "🟩 Buy", "Sell": "🟥 Sell"}
                ).fillna(tx_df["transaction_type"])

            col_rename = {
                "date": "Дата",
                "isin": "ISIN",
                "asset_name": "Актив",
                "asset_category": "Категория",
                "transaction_type": "Тип",
                "total_value": "Сумма",
            }
            tx_df = tx_df.rename(columns=col_rename)
            display_cols = [c for c in col_rename.values() if c in tx_df.columns]
            st.dataframe(tx_df[display_cols], use_container_width=True, hide_index=True)


def main():
    _init_state()

    if st.session_state["page"] == "list":
        show_client_list()
    elif st.session_state["page"] == "card":
        cid = st.session_state.get("selected_client")
        if cid:
            show_client_card(cid)
        else:
            st.session_state["page"] = "list"
            st.rerun()


if __name__ == "__main__":
    main()

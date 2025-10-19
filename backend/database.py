import sqlite3
import json
from datetime import datetime
from contextlib import contextmanager
import os

# Always use absolute path so the same DB file is used across reloads/processes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_NAME = os.path.join(BASE_DIR, 'forex_signals.db')

@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_database():
    """Initialize the database with required tables"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Create signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                confidence TEXT,
                entry_price REAL,
                stop_loss REAL,
                take_profit_1 REAL,
                take_profit_2 REAL,
                take_profit_3 REAL,
                lot_size REAL,
                risk_amount REAL,
                potential_profit_1 REAL,
                potential_profit_2 REAL,
                potential_profit_3 REAL,
                status TEXT DEFAULT 'active',
                pips REAL DEFAULT 0,
                performance_outcome TEXT, -- Successful, Failed, Neutral
                analysis_summary TEXT,
                detailed_explanation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                signal_id TEXT
            )
        ''')

        # Ensure upgraded databases also have signal_id column
        cursor.execute("PRAGMA table_info(signals)")
        columns = [row[1] for row in cursor.fetchall()]
        if "signal_id" not in columns:
            cursor.execute("ALTER TABLE signals ADD COLUMN signal_id TEXT")

        # Phase A indicators
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS indicators_phase_a (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                rsi REAL,
                ema9 REAL,
                ema21 REAL,
                macd TEXT,
                bbands TEXT,
                atr REAL,
                obv REAL,
                volume TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Phase B indicators
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS indicators_phase_b (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                rsi REAL,
                ema9 REAL,
                ema21 REAL,
                macd TEXT,
                bbands TEXT,
                atr REAL,
                obv REAL,
                sentiment_score REAL,
                volume TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Cache: OHLC candles (JSON blob per timeframe fetch)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS candles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data_json TEXT NOT NULL
            )
        ''')

        # Cache: indicators by kind (RSI/MACD/EMA/BBANDS/ATR/VWAP/OBV/LuxAlgo)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                kind TEXT NOT NULL,
                computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data_json TEXT NOT NULL
            )
        ''')

        # Cache: news articles with sentiment
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                published_at TIMESTAMP NOT NULL,
                headline TEXT NOT NULL,
                sentiment TEXT,
                source TEXT,
                url TEXT
            )
        ''')

        # Create signal chats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signal_chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER NOT NULL,
                role TEXT NOT NULL, -- 'user' or 'assistant'
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (signal_id) REFERENCES signals (id)
            )
        ''')

        # Create general-purpose chat table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS general_chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL, -- 'user' or 'assistant'
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Migration: Add new indicator columns to indicators_phase_a if they don't exist
        cursor.execute("PRAGMA table_info(indicators_phase_a)")
        phase_a_columns = [row[1] for row in cursor.fetchall()]
        if "volume" not in phase_a_columns:
            cursor.execute("ALTER TABLE indicators_phase_a ADD COLUMN volume TEXT")
            print("Added volume column to indicators_phase_a")
        if "macd" not in phase_a_columns:
            cursor.execute("ALTER TABLE indicators_phase_a ADD COLUMN macd TEXT")
            print("Added macd column to indicators_phase_a")
        if "bbands" not in phase_a_columns:
            cursor.execute("ALTER TABLE indicators_phase_a ADD COLUMN bbands TEXT")
            print("Added bbands column to indicators_phase_a")
        if "atr" not in phase_a_columns:
            cursor.execute("ALTER TABLE indicators_phase_a ADD COLUMN atr REAL")
            print("Added atr column to indicators_phase_a")
        if "obv" not in phase_a_columns:
            cursor.execute("ALTER TABLE indicators_phase_a ADD COLUMN obv REAL")
            print("Added obv column to indicators_phase_a")
        
        # Migration: Add new indicator columns to indicators_phase_b if they don't exist
        cursor.execute("PRAGMA table_info(indicators_phase_b)")
        phase_b_columns = [row[1] for row in cursor.fetchall()]
        if "volume" not in phase_b_columns:
            cursor.execute("ALTER TABLE indicators_phase_b ADD COLUMN volume TEXT")
            print("Added volume column to indicators_phase_b")
        if "bbands" not in phase_b_columns:
            cursor.execute("ALTER TABLE indicators_phase_b ADD COLUMN bbands TEXT")
            print("Added bbands column to indicators_phase_b")
        if "atr" not in phase_b_columns:
            cursor.execute("ALTER TABLE indicators_phase_b ADD COLUMN atr REAL")
            print("Added atr column to indicators_phase_b")
        if "obv" not in phase_b_columns:
            cursor.execute("ALTER TABLE indicators_phase_b ADD COLUMN obv REAL")
            print("Added obv column to indicators_phase_b")
        
        conn.commit()
        print("Database initialized successfully")

def save_signal(signal_data):
    """Save a signal to the database"""
    if not signal_data:
        raise ValueError("signal_data payload is required")

    recommendation = (
        signal_data.get('recommendation')
        or signal_data.get('signal')
        or 'NO TRADE'
    )

    signal_id = (
        signal_data.get('signal_id')
        or signal_data.get('id')
        or signal_data.get('database_id')
    )

    confidence = signal_data.get('confidence')
    if confidence is None:
        confidence_str = None
    elif isinstance(confidence, (int, float)):
        confidence_str = f"{confidence}"
    else:
        confidence_str = str(confidence)

    entry_price = signal_data.get('entry_price') or signal_data.get('entry_point')
    take_profit_1 = signal_data.get('take_profit_1') or signal_data.get('take_profit')
    take_profit_2 = signal_data.get('take_profit_2') or signal_data.get('secondary_take_profit')
    take_profit_3 = signal_data.get('take_profit_3') or signal_data.get('tertiary_take_profit')

    lot_size = signal_data.get('lot_size')
    risk_amount = signal_data.get('risk_amount')
    potential_profit_1 = signal_data.get('potential_profit_1') or signal_data.get('potential_profit')
    potential_profit_2 = signal_data.get('potential_profit_2') or signal_data.get('secondary_potential_profit')
    potential_profit_3 = signal_data.get('potential_profit_3') or signal_data.get('tertiary_potential_profit')

    status = signal_data.get('status')
    if not status:
        status = 'active' if recommendation != 'NO TRADE' else 'expired'

    analysis_summary_data = (
        signal_data.get('analysis_summary')
        or signal_data.get('reason')
        or {}
    )
    detailed_explanation_data = (
        signal_data.get('detailed_explanation')
        or signal_data.get('commentary')
        or signal_data.get('bias')
        or {}
    )

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO signals (
                pair, recommendation, confidence, entry_price, stop_loss,
                take_profit_1, take_profit_2, take_profit_3,
                lot_size, risk_amount,
                potential_profit_1, potential_profit_2, potential_profit_3,
                status, analysis_summary, detailed_explanation, signal_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal_data.get('pair'),
            recommendation,
            confidence_str,
            entry_price,
            signal_data.get('stop_loss'),
            take_profit_1,
            take_profit_2,
            take_profit_3,
            lot_size,
            risk_amount,
            potential_profit_1,
            potential_profit_2,
            potential_profit_3,
            status,
            json.dumps(analysis_summary_data),
            json.dumps(detailed_explanation_data),
            signal_id
        ))

        conn.commit()
        return cursor.lastrowid

def get_all_signals():
    """Get all signals from the database"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM signals ORDER BY created_at DESC')
        rows = cursor.fetchall()
        
        signals = []
        for row in rows:
            try:
                signal = dict(row)
                # Parse JSON fields
                signal['analysis_summary'] = json.loads(signal['analysis_summary']) if signal['analysis_summary'] else {}
                signal['detailed_explanation'] = json.loads(signal['detailed_explanation']) if signal['detailed_explanation'] else {}
                signals.append(signal)
            except Exception as e:
                print(f"[db-error] Skipping corrupt signal record id={row['id'] if 'id' in row else 'N/A'}: {e}")
                continue
        
        return signals

def get_all_pairs():
    """Get a list of all unique pairs from the signals table."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT pair FROM signals ORDER BY pair ASC')
        return [row['pair'] for row in cursor.fetchall()]

def save_chat_message(signal_id: int, role: str, content: str) -> int:
    """Save a chat message to the database."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO signal_chats (signal_id, role, content) VALUES (?, ?, ?)',
            (signal_id, role, content)
        )
        conn.commit()
        return cursor.lastrowid

def get_chat_history(signal_id: int) -> list[dict]:
    """Retrieve chat history for a given signal ID from the last 30 days only."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            '''
            SELECT role, content, timestamp
            FROM signal_chats
            WHERE signal_id = ?
              AND timestamp >= datetime('now', '-30 days')
            ORDER BY timestamp ASC
            ''',
            (signal_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

def delete_old_chat_messages(days: int = 30) -> int:
    """Delete chat messages older than the specified number of days. Returns rows deleted."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM signal_chats WHERE timestamp < datetime('now', ?)",
            (f'-{int(days)} days',)
        )
        conn.commit()
        return cursor.rowcount

def save_general_chat_message(role: str, content: str) -> int:
    """Save a message to the general_chats table."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO general_chats (role, content) VALUES (?, ?)',
            (role, content)
        )
        conn.commit()
        return cursor.lastrowid

def get_general_chat_history() -> list[dict]:
    """Retrieve the last 30 days of general chat history."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            '''
            SELECT role, content, timestamp
            FROM general_chats
            WHERE timestamp >= datetime('now', '-30 days')
            ORDER BY timestamp ASC
            '''
        )
        return [dict(row) for row in cursor.fetchall()]

def delete_old_general_chat_messages(days: int = 30) -> int:
    """Delete general chat messages older than N days."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM general_chats WHERE timestamp < datetime('now', ?)",
            (f'-{int(days)} days',)
        )
        conn.commit()
        return cursor.rowcount

def get_signals_by_status(status):
    """Get signals filtered by status"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM signals WHERE status = ? ORDER BY created_at DESC', (status,))
        rows = cursor.fetchall()
        
        signals = []
        for row in rows:
            try:
                signal = dict(row)
                signal['analysis_summary'] = json.loads(signal['analysis_summary']) if signal['analysis_summary'] else {}
                signal['detailed_explanation'] = json.loads(signal['detailed_explanation']) if signal['detailed_explanation'] else {}
                signals.append(signal)
            except Exception as e:
                print(f"[db-error] Skipping corrupt signal record id={row['id'] if 'id' in row else 'N/A'}: {e}")
                continue
        
        return signals

def get_active_signals():
    """Get all active signals"""
    return get_signals_by_status('active')

def update_signal_status(signal_id, status, pips=None):
    """Update a signal's status and pips"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        if pips is not None:
            cursor.execute('''
                UPDATE signals 
                SET status = ?, pips = ?, updated_at = CURRENT_TIMESTAMP 
                WHERE id = ?
            ''', (status, pips, signal_id))
        else:
            cursor.execute('''
                UPDATE signals 
                SET status = ?, updated_at = CURRENT_TIMESTAMP 
                WHERE id = ?
            ''', (status, signal_id))
        
        conn.commit()

def get_performance_stats():
    """Get overall performance statistics"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Total signals
        cursor.execute('SELECT COUNT(*) as count FROM signals')
        total_signals = cursor.fetchone()['count']
        
        # Status breakdown
        cursor.execute('SELECT status, COUNT(*) as count FROM signals GROUP BY status')
        status_counts = {row['status']: row['count'] for row in cursor.fetchall()}
        
        # Win rate calculation (hit_tp = wins)
        hit_tp_count = status_counts.get('hit_tp', 0)
        hit_sl_count = status_counts.get('hit_sl', 0)
        closed_signals = hit_tp_count + hit_sl_count
        win_rate = (hit_tp_count / closed_signals * 100) if closed_signals > 0 else 0
        
        # Net pips
        cursor.execute('SELECT SUM(pips) as total_pips FROM signals WHERE status IN ("hit_tp", "hit_sl")')
        result = cursor.fetchone()
        net_pips = result['total_pips'] if result['total_pips'] else 0
        
        # Average confidence
        cursor.execute('SELECT AVG(CAST(SUBSTR(confidence, 1, 1) AS INTEGER)) as avg_conf FROM signals WHERE confidence IS NOT NULL')
        result = cursor.fetchone()
        avg_confidence = result['avg_conf'] if result['avg_conf'] else 0
        
        return {
            'total_signals': total_signals,
            'win_rate': round(win_rate, 1),
            'net_pips': round(net_pips, 1),
            'avg_confidence': round(avg_confidence, 1),
            'status_counts': status_counts,
            'active': status_counts.get('active', 0),
            'hit_tp': hit_tp_count,
            'hit_sl': hit_sl_count,
            'expired': status_counts.get('expired', 0)
        }

def get_pair_performance():
    """Get performance breakdown by currency pair"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                pair,
                COUNT(*) as total_signals,
                SUM(CASE WHEN status = 'hit_tp' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN status = 'hit_sl' THEN 1 ELSE 0 END) as losses,
                SUM(pips) as total_pips
            FROM signals
            WHERE status IN ('hit_tp', 'hit_sl')
            GROUP BY pair
            ORDER BY total_pips DESC
        ''')
        
        pairs = []
        for row in cursor.fetchall():
            total = row['wins'] + row['losses']
            win_rate = (row['wins'] / total * 100) if total > 0 else 0
            
            pairs.append({
                'pair': row['pair'],
                'total_signals': row['total_signals'],
                'wins': row['wins'],
                'losses': row['losses'],
                'win_rate': round(win_rate, 1),
                'total_pips': round(row['total_pips'], 1)
            })
        
        return pairs

# Initialize database when module is imported
init_database()

# -------------------- Cache Helpers --------------------
def cache_candles(pair: str, timeframe: str, candles_payload: dict) -> None:
    """Store latest candles snapshot as JSON for a pair/timeframe."""
    with get_db() as conn:
        conn.execute(
            'INSERT INTO candles(pair, timeframe, data_json) VALUES (?, ?, ?)',
            (pair, timeframe, json.dumps(candles_payload)),
        )
        conn.commit()

def cache_indicator(pair: str, timeframe: str, kind: str, data: dict) -> None:
    """Store indicator values for a pair/timeframe/kind."""
    with get_db() as conn:
        conn.execute(
            'INSERT INTO indicators(pair, timeframe, kind, data_json) VALUES (?, ?, ?, ?)',
            (pair, timeframe, kind.upper(), json.dumps(data)),
        )
        conn.commit()

def cache_news(pair: str, headline: str, sentiment: str | None, published_at: str, source: str | None = None, url: str | None = None) -> None:
    """Store one news item with sentiment; published_at should be ISO or RFC3339."""
    with get_db() as conn:
        conn.execute(
            'INSERT INTO news_articles(pair, published_at, headline, sentiment, source, url) VALUES (?, ?, ?, ?, ?, ?)',
            (pair, published_at, headline, sentiment, source, url),
        )
        conn.commit()

def load_recent_news(pair: str, lookback_hours: int = 6) -> list[dict]:
    with get_db() as conn:
        cur = conn.execute(
            '''SELECT pair, published_at, headline, sentiment, source, url
               FROM news_articles
               WHERE pair = ? AND datetime(published_at) >= datetime('now', ?)
               ORDER BY datetime(published_at) DESC
            ''',
            (pair, f'-{lookback_hours} hours'),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]

def get_signal(signal_id: int) -> dict | None:
    """Get a single signal by its ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM signals WHERE id = ?', (signal_id,))
        row = cursor.fetchone()
        if row:
            signal = dict(row)
            signal['analysis_summary'] = json.loads(signal['analysis_summary']) if signal['analysis_summary'] else {}
            signal['detailed_explanation'] = json.loads(signal['detailed_explanation']) if signal['detailed_explanation'] else {}
            return signal
        return None

def cleanup_old_cache(days: int = 7) -> None:
    """Delete cache rows older than N days."""
    with get_db() as conn:
        for table, col in (
            ('candles', 'fetched_at'), 
            ('indicators', 'computed_at'), 
            ('news_articles', 'published_at'),
            ('indicators_phase_a', 'timestamp'),
            ('indicators_phase_b', 'timestamp')
        ):
            conn.execute(
                f"DELETE FROM {table} WHERE datetime({col}) < datetime('now', ?)",
                (f'-{days} days',),
            )
        conn.commit()

def save_phase_a_indicators(pair: str, timeframe: str, rsi: float, ema9: float, ema21: float, 
                           macd: dict, bbands: dict, atr: float, obv: float, volume: dict) -> None:
    with get_db() as conn:
        conn.execute(
            '''INSERT INTO indicators_phase_a(pair, timeframe, rsi, ema9, ema21, macd, bbands, atr, obv, volume) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (pair, timeframe, rsi, ema9, ema21, json.dumps(macd), json.dumps(bbands), atr, obv, json.dumps(volume))
        )
        conn.commit()

def save_phase_b_indicators(pair: str, timeframe: str, rsi: float, ema9: float, ema21: float, 
                           macd: dict, bbands: dict, atr: float, obv: float, sentiment: float, volume: dict) -> None:
    with get_db() as conn:
        conn.execute(
            '''INSERT INTO indicators_phase_b(pair, timeframe, rsi, ema9, ema21, macd, bbands, atr, obv, sentiment_score, volume) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (pair, timeframe, rsi, ema9, ema21, json.dumps(macd), json.dumps(bbands), atr, obv, sentiment, json.dumps(volume))
        )
        conn.commit()

def update_signal_performance(signal_id: int, outcome: str) -> None:
    """Update the performance_outcome of a signal."""
    with get_db() as conn:
        conn.execute(
            'UPDATE signals SET performance_outcome = ? WHERE id = ?',
            (outcome, signal_id)
        )
        conn.commit()

def get_phase_a_indicators(pair: str) -> dict:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''SELECT timeframe, rsi, ema9, ema21, macd, bbands, atr, obv, volume 
                          FROM indicators_phase_a WHERE pair = ? ORDER BY timestamp DESC''', (pair,))
        rows = cursor.fetchall()
        data = {}
        for row in rows:
            if row['timeframe'] not in data:
                row_dict = dict(row)
                row_dict['macd'] = json.loads(row_dict['macd']) if row_dict.get('macd') else {}
                row_dict['bbands'] = json.loads(row_dict['bbands']) if row_dict.get('bbands') else {}
                row_dict['volume'] = json.loads(row_dict['volume']) if row_dict['volume'] else {}
                data[row['timeframe']] = row_dict
        return data

def get_phase_b_indicators(pair: str) -> dict:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''SELECT timeframe, rsi, ema9, ema21, macd, bbands, atr, obv, volume 
                          FROM indicators_phase_b WHERE pair = ? ORDER BY timestamp DESC''', (pair,))
        rows = cursor.fetchall()
        data = {}
        for row in rows:
            if row['timeframe'] not in data:
                row_dict = dict(row)
                row_dict['macd'] = json.loads(row_dict['macd']) if row_dict.get('macd') else {}
                row_dict['bbands'] = json.loads(row_dict['bbands']) if row_dict.get('bbands') else {}
                row_dict['volume'] = json.loads(row_dict['volume']) if row_dict['volume'] else {}
                data[row['timeframe']] = row_dict
        return data
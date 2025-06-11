from tensorflow.keras.layers import Input, LSTM, MultiHeadAttention, Concatenate, LayerNormalization, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

def create_combined_model(temporal_shape, news_shape):
    # Này là input sequence (giá bla bla, thời gian bla bla)
    temporal_input = Input(shape=temporal_shape, name='Temporal_Input')  

    # LSTM layer
    lstm_out, forward_h, forward_c = LSTM(50, activation='relu', return_sequences=True, return_state=True)(temporal_input)

    # Multi-Head Attention + causal mask
    attention = MultiHeadAttention(num_heads=2, key_dim=64, name='MultiHeadAttention')
    attention_out = attention(lstm_out, lstm_out, use_causal_mask=True)

    # Fusion LSTM + Attention output
    combined_temporal = Concatenate(name='Temporal_Concat')([lstm_out, attention_out])

    # Normalize
    norm = LayerNormalization(epsilon=1e-6, name='LayerNorm')(combined_temporal)

    # Flatten temporal feature
    flat_temporal = Flatten(name='Flatten_Temporal')(norm)

    # News 
    news_input = Input(shape=(news_shape,), name='News_Input')  # (num_features,)
    news_dense = Dense(1, activation='relu', name='News_Dense_1')(news_input)
  

    #  Fusion temporal and news features
    fusion = Concatenate(name='Fusion')([flat_temporal, news_dense])
    output = Dense(1, activation='linear', name='Output')(fusion)

    # Model configuration
    model = Model(inputs=[temporal_input, news_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber(delta=1.0))
    return model




{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "c89b7974-e091-4032-8707-8cc6de1a8c6e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>teks</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Produk ini sangat bagus</td>\n",
       "      <td>positif</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Saya kecewa dengan kualitasnya</td>\n",
       "      <td>negatif</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Layanan pelanggan sangat membantu</td>\n",
       "      <td>positif</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Saya tidak akan beli lagi</td>\n",
       "      <td>negatif</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Barang datang tepat waktu</td>\n",
       "      <td>positif</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                teks    label\n",
       "0            Produk ini sangat bagus  positif\n",
       "1     Saya kecewa dengan kualitasnya  negatif\n",
       "2  Layanan pelanggan sangat membantu  positif\n",
       "3          Saya tidak akan beli lagi  negatif\n",
       "4          Barang datang tepat waktu  positif"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# Contoh dataset sentimen sederhana (120 baris)\n",
    "data = {\n",
    "    \"teks\": [\n",
    "        \"Produk ini sangat bagus\", \"Saya kecewa dengan kualitasnya\", \"Layanan pelanggan sangat membantu\",\n",
    "        \"Saya tidak akan beli lagi\", \"Barang datang tepat waktu\", \"Pengiriman sangat lambat\",\n",
    "        \"Harga terjangkau dan kualitas oke\", \"Tidak sesuai dengan deskripsi\", \"Sangat puas dengan pembelian ini\",\n",
    "        \"Ini pengalaman belanja yang buruk\"\n",
    "    ] * 12,  # 10 x 12 = 120 data\n",
    "    \"label\": [\n",
    "        \"positif\", \"negatif\", \"positif\", \"negatif\", \"positif\",\n",
    "        \"negatif\", \"positif\", \"negatif\", \"positif\", \"negatif\"\n",
    "    ] * 12\n",
    "}\n",
    "\n",
    "df = pd.DataFrame(data)\n",
    "df.to_csv(\"dataset_sentimen.csv\", index=False)\n",
    "df.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "e624775f-853c-4e90-85b0-f5786fd3c6f9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Akurasi model: 1.0\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.pipeline import make_pipeline\n",
    "\n",
    "# Load dataset\n",
    "df = pd.read_csv(\"dataset_sentimen.csv\")\n",
    "\n",
    "# Split data\n",
    "X_train, X_test, y_train, y_test = train_test_split(df['teks'], df['label'], test_size=0.2, random_state=42)\n",
    "\n",
    "# Pipeline model\n",
    "model = make_pipeline(CountVectorizer(), MultinomialNB())\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Evaluasi\n",
    "print(\"Akurasi model:\", model.score(X_test, y_test))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "51f9705f-3899-4539-b6e5-45c24801d1b9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ea27a6ce6cd64fe0957e9ba31a7fc283",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Textarea(value='Tulis opini Anda di sini...', description='Teks:', layout=Layout(height='100px', width='100%')…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ec50d483d99a4b32a23864e8a16a0386",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Button(description='Prediksi Sentimen', style=ButtonStyle())"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4781a36d625745bf9560acd623bc7fdf",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Output()"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import ipywidgets as widgets\n",
    "from IPython.display import display\n",
    "\n",
    "# Input teks dari pengguna\n",
    "input_box = widgets.Textarea(\n",
    "    value='Tulis opini Anda di sini...',\n",
    "    placeholder='Tulis opini Anda di sini...',\n",
    "    description='Teks:',\n",
    "    layout=widgets.Layout(width='100%', height='100px')\n",
    ")\n",
    "\n",
    "# Tombol\n",
    "button = widgets.Button(description=\"Prediksi Sentimen\")\n",
    "\n",
    "# Output\n",
    "output = widgets.Output()\n",
    "\n",
    "# Fungsi prediksi\n",
    "def on_button_clicked(b):\n",
    "    with output:\n",
    "        output.clear_output()\n",
    "        user_input = input_box.value\n",
    "        prediksi = model.predict([user_input])[0]\n",
    "        print(f\"Sentimen: {prediksi.upper()}\")\n",
    "\n",
    "# Hubungkan tombol dengan fungsi\n",
    "button.on_click(on_button_clicked)\n",
    "\n",
    "# Tampilkan semuanya\n",
    "display(input_box, button, output)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "2ff90986-c433-4650-bf41-03074115842c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "* Running on local URL:  http://127.0.0.1:7860\n",
      "* To create a public link, set `share=True` in `launch()`.\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div><iframe src=\"http://127.0.0.1:7860/\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": []
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.pipeline import make_pipeline\n",
    "import gradio as gr\n",
    "\n",
    "# ======== Dataset =========\n",
    "data = {\n",
    "    \"teks\": [\n",
    "        \"Produk ini sangat bagus\", \"Saya kecewa dengan kualitasnya\", \"Layanan pelanggan sangat membantu\",\n",
    "        \"Saya tidak akan beli lagi\", \"Barang datang tepat waktu\", \"Pengiriman sangat lambat\",\n",
    "        \"Harga terjangkau dan kualitas oke\", \"Tidak sesuai dengan deskripsi\", \"Sangat puas dengan pembelian ini\",\n",
    "        \"Ini pengalaman belanja yang buruk\"\n",
    "    ] * 12,  # 120 data\n",
    "    \"label\": [\n",
    "        \"positif\", \"negatif\", \"positif\", \"negatif\", \"positif\",\n",
    "        \"negatif\", \"positif\", \"negatif\", \"positif\", \"negatif\"\n",
    "    ] * 12\n",
    "}\n",
    "df = pd.DataFrame(data)\n",
    "\n",
    "# ======== Training =========\n",
    "X_train, X_test, y_train, y_test = train_test_split(df['teks'], df['label'], test_size=0.2, random_state=42)\n",
    "model = make_pipeline(CountVectorizer(), MultinomialNB())\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# ======== Fungsi Prediksi =========\n",
    "def prediksi_sentimen(teks):\n",
    "    hasil = model.predict([teks])[0]\n",
    "    return f\"Sentimen: {hasil.upper()}\"\n",
    "\n",
    "# ======== Web Interface =========\n",
    "app = gr.Interface(\n",
    "    fn=prediksi_sentimen,\n",
    "    inputs=gr.Textbox(lines=4, placeholder=\"Tulis opini Anda di sini...\"),\n",
    "    outputs=\"text\",\n",
    "    title=\"Klasifikasi Sentimen\",\n",
    "    description=\"Masukkan teks opini, dan sistem akan memprediksi apakah sentimennya POSITIF atau NEGATIF.\"\n",
    ")\n",
    "\n",
    "# ======== Jalankan Web App =========\n",
    "app.launch()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5ea4f6ca-f00c-4079-9e72-193f159cb962",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

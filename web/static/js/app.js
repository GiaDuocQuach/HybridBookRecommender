// static/js/app.js
document.addEventListener('DOMContentLoaded', function() {
  const resultsContainer = document.getElementById('results');
  const btnSearch = document.getElementById('btn-search');
  const keywordInput = document.getElementById('keyword');
  const suggestionsList = document.getElementById('suggestions');
  let currentKeyword = '';

  function renderResults(items) {
    resultsContainer.innerHTML = '';
    if (!items.length) {
      resultsContainer.innerHTML = '<div class="no-result">Không tìm thấy kết quả phù hợp.</div>';
      return;
    }
    items.forEach(b => {
      const card = document.createElement('div');
      card.className = 'book-card';
      card.setAttribute('data-book-idx', b.book_idx);

      // Badge popularity
      const badge = document.createElement('div');
      badge.className = `card-badge ${b.popularity_level.toLowerCase()}`;
      badge.textContent = b.popularity_level;
      card.appendChild(badge);

      // Image
      const img = document.createElement('img');
      img.src = b.image_url;
      img.alt = b.title;
      card.appendChild(img);

      // Body
      const body = document.createElement('div');
      body.className = 'card-body';

      // Highlight keyword in title
      function highlight(text, keyword) {
        if (!keyword) return text;
        const re = new RegExp(`(${keyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
        return text.replace(re, '<span class="highlight">$1</span>');
      }

      const h4 = document.createElement('h4');
      h4.innerHTML = highlight(b.title, currentKeyword);
      body.appendChild(h4);

      const pAuthor = document.createElement('p');
      pAuthor.textContent = b.author;
      body.appendChild(pAuthor);

      const pPrice = document.createElement('p');
      pPrice.textContent = `Giá: ${b.price}`;
      body.appendChild(pPrice);

      const pSold = document.createElement('p');
      pSold.textContent = `Đã bán: ${b.sold_count}`;
      body.appendChild(pSold);

      const pStars = document.createElement('p');
      pStars.className = 'stars';
      pStars.textContent = `${b.stars} (${b.rating})`;
      body.appendChild(pStars);

      card.appendChild(body);

      // Khi click vào card: gửi interaction rồi redirect detail
      card.addEventListener('click', function() {
        const bookIdx = this.getAttribute('data-book-idx');
        fetch('/click', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({book_idx: bookIdx})
        })
        .finally(() => {
          window.location.href = `/book/${bookIdx}`;
        });
      });

      resultsContainer.appendChild(card);
    });
  }

  function showLoading() {
    resultsContainer.innerHTML = '<div class="loading">Đang tìm kiếm...</div>';
    btnSearch.disabled = true;
  }

  function hideLoading() {
    btnSearch.disabled = false;
  }

  function searchBooks(query) {
    showLoading();
    fetch('/recommend', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(query)
    })
    .then(r => r.json())
    .then(data => {
      hideLoading();
      if (data.error) {
        resultsContainer.innerHTML = `<div class="no-result">Lỗi: ${data.error}</div>`;
      } else {
        renderResults(data.results || []);
      }
    })
    .catch(err => {
      resultsContainer.innerHTML = `<div class="no-result">Lỗi kết nối: ${err}</div>`;
      hideLoading();
    });
  }

  btnSearch.onclick = function() {
    const category = document.getElementById('category').value;
    const price_ranges = Array.from(document.querySelectorAll('input[name="price_ranges"]:checked')).map(e => e.value);
    const pop_level = document.querySelector('input[name="pop_levels"]:checked')?.value;
    const rating_bin = document.querySelector('input[name="rating_bins"]:checked')?.value;
    const keyword = keywordInput.value.trim();
    currentKeyword = keyword;

    const payload = {
      category,
      price_ranges,
      pop_levels: pop_level ? [pop_level] : [],
      rating_bins: rating_bin ? [rating_bin] : [],
      keyword
    };
    searchBooks(payload);
  };

  // Autocomplete
  let debounceTimer;
  keywordInput.addEventListener('input', function(e) {
    const val = this.value.trim();
    clearTimeout(debounceTimer);
    if (!val) {
      suggestionsList.innerHTML = '';
      return;
    }
    debounceTimer = setTimeout(() => {
      fetch(`/suggest?prefix=${encodeURIComponent(val)}`)
        .then(r => r.json())
        .then(data => {
          suggestionsList.innerHTML = '';
          data.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            li.onclick = () => {
              keywordInput.value = item;
              suggestionsList.innerHTML = '';
            };
            suggestionsList.appendChild(li);
          });
        });
    }, 300);
  });

  document.addEventListener('click', function(e) {
    if (!keywordInput.contains(e.target) && !suggestionsList.contains(e.target)) {
      suggestionsList.innerHTML = '';
    }
  });

  // Mặc định message khi vào trang index (được render trong template)
});

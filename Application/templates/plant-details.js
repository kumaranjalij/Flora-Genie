const postContainer= document.querySelector('.card-container');

const postMethods = ()=>{
    plantResultObj.map((postData) => {
        const postElement = document.createElement('div');
        postElement.classList.add('card');
        postElement.innerHTML = `
        <h3 class="card-heading"> ${postData.heading} </h3>
        <p class="card-body"> ${postData.body}<p>
        `

        postContainer.appendChild(postElement)

        console.log(postData)
    })
}

postMethods()



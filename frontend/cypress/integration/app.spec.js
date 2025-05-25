describe('App', () => {
  beforeEach(() => {
    cy.visit('/')
  })

  it('should display the main page', () => {
    cy.get('h1').should('contain', 'EmberHunter')
  })

  it('should have upload buttons', () => {
    cy.get('button').should('contain', '上传热成像图片')
    cy.get('button').should('contain', '上传红外图片')
  })

  it('should have process button', () => {
    cy.get('button').should('contain', '开始处理')
  })
}) 